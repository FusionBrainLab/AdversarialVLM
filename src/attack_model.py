import sys
sys.path = [p for p in sys.path if p != '/home/jovyan/.imgenv-razzhigaev-small-1-0/lib/python3.7/site-packages'] 

from typing import Optional, Union
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
# from transformers import AutoProcessor, AutoModelForCausalLM
import os
import argparse
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import random  # Added import for random sampling
import json

# Conditional import for Aim
try:
    from aim import Run
    try:
        from aim import Image as AimImage
        AIM_IMAGE_AVAILABLE = True
    except ImportError:
        AIM_IMAGE_AVAILABLE = False
        AimImage = None
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    AIM_IMAGE_AVAILABLE = False
    print("Warning: Aim not installed. Install with: pip install aim")
    Run = None
    AimImage = None

from processors import load_components
from train_test import run_model_test

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import GaussianBlur # Additional regularization for noise

class ExperimentLogger:
    """Universal wrapper for both Aim and TensorBoard logging."""
    
    def __init__(self, exp_name: str, config: dict, use_tensorboard: bool = False, track_artifacts: bool = False):
        self.exp_name = exp_name
        self.config = config
        self.use_tensorboard = use_tensorboard
        self.track_artifacts = track_artifacts
        
        if use_tensorboard:
            # Initialize TensorBoard
            self.writer = SummaryWriter(log_dir=os.path.join("./runs", exp_name))
            self.aim_run = None
            print(f"Using TensorBoard for experiment: {exp_name}")
        else:
            # Initialize Aim (default)
            if not AIM_AVAILABLE:
                raise ImportError("Aim is not available. Install with 'pip install aim' or use --use_tensorboard flag")
            
            self.aim_run = Run(
                repo='./aim_repo',  # Local Aim repository
                experiment=exp_name
            )
            # Log config parameters
            for key, value in config.items():
                self.aim_run[key] = value
            
            self.writer = None
            print(f"Using Aim for experiment: {exp_name}")
        
        print(f"Artifact tracking (images/tensors): {'enabled' if track_artifacts else 'disabled'}")
    
    def add_scalar(self, name: str, value: float, step: int):
        """Log scalar metric."""
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar(name, value, step)
        elif self.aim_run:
            self.aim_run.track(value, name=name, step=step)
    
    def add_image(self, name: str, img_tensor: torch.Tensor, step: int):
        """Log image (only if artifacts tracking is enabled)."""
        if not self.track_artifacts:
            return  # Skip image logging if artifacts tracking is disabled
            
        if self.use_tensorboard and self.writer:
            self.writer.add_image(name, img_tensor, step)
        elif self.aim_run:
            try:
                # Convert tensor to numpy array for Aim
                if len(img_tensor.shape) == 4:  # Batch dimension
                    img_tensor = img_tensor[0]
                
                # Ensure tensor is in [0,1] range and convert to numpy
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                
                # Convert to numpy array (HWC format for Aim)
                if img_tensor.shape[0] == 3:  # CHW -> HWC
                    img_array = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
                else:
                    img_array = img_tensor.cpu().detach().numpy()
                
                # Ensure values are in [0,1] range
                img_array = np.clip(img_array, 0, 1)
                
                # Track as numpy array (Aim can handle this)
                if AIM_IMAGE_AVAILABLE:
                    aim_image = AimImage(img_array)
                    self.aim_run.track(aim_image, name=name, step=step)
                else:
                    # Fallback: try to track as raw numpy array
                    self.aim_run.track(img_array, name=name, step=step)
                
            except Exception as e:
                print(f"Warning: Failed to log image {name} to Aim: {e}")
    
    def close(self):
        """Close logger."""
        if self.use_tensorboard and self.writer:
            self.writer.close()
        elif self.aim_run:
            self.aim_run.close()

def setup_device():
    """Setup computing device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directory(exp_name, base_path="./runs"):
    """Creates a directory for experiment logs and outputs."""
    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def save_checkpoint(image: Image.Image, tensor: torch.Tensor, path: str, iteration: int):
    """Saves the current image checkpoint."""
    image.save(os.path.join(path, f"optimized_image_iter_{iteration}.png"))
    tensor.cpu().detach().numpy().astype(np.float32).tofile(os.path.join(path, f"optimized_image_iter_{iteration}.bin"))

def initialize_experiment_logger(exp_name, config, use_tensorboard=False, track_artifacts=False):
    """Initialize experiment logger (Aim by default, TensorBoard if specified)."""
    return ExperimentLogger(exp_name, config, use_tensorboard, track_artifacts)

def log_metrics(
        logger: ExperimentLogger,
        iteration: int,
        loss: torch.Tensor,
        final_image: Image.Image,
        final_tensor: torch.Tensor,
        save_steps: int,
        additional_log: Optional[dict] = None
    ):
    """Logs metrics and images to the experiment logger."""

    if iteration % save_steps == 0:  # Log images and model output every `save_steps` iterations
        # Convert PIL image to tensor for logging
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(final_image)
        logger.add_image(f"optimized_image_iter_{iteration}", img_tensor, iteration)
        # log x+x_0
        logger.add_image(f"optimized_tensor_iter_{iteration}", final_tensor, iteration)
    
    if additional_log is not None:
        for key, value in additional_log.items():
            logger.add_scalar(key, value, iteration)
        logger.add_scalar("loss", loss.item(), iteration)

def create_mask(mask_type, mask_size, image_shape, device):
    """Creates a mask tensor based on the specified mask_type and mask_size."""
    mask = torch.zeros(image_shape).to(device)
    C, H, W = image_shape
    if mask_type == 'corner':
        n = mask_size
        mask[:, :n, :n] = 1.0
    elif mask_type == 'bottom_lines':
        k = mask_size
        mask[:, -k:, :] = 1.0
    elif mask_type == 'random_square':
        n = mask_size
        # Randomly select top-left corner
        i = random.randint(0, H - n)
        j = random.randint(0, W - n)
        mask[:, i:i+n, j:j+n] = 1.0
    else:
        mask = torch.ones(image_shape).to(device)
    return mask

def image_fit_loss(
        x_0: torch.Tensor, 
        x: torch.Tensor, 
        lower_bound, 
        upper_bound, 
        center_force = 0.9
    ):
    # Считаем сумму
    x_sum = (x_0 + x)
    
    lower_bound = torch.zeros(x_sum.shape).to(x_sum.device)
    upper_bound = torch.ones(x_sum.shape).to(x_sum.device)
    
    # Рассчитываем штраф за выход за границы
    lower_penalty = torch.relu(center_force*lower_bound - x_sum)  # если x_sum меньше нижней границы
    upper_penalty = torch.relu(x_sum - center_force*upper_bound)  # если x_sum больше верхней границы
    
    # Используем MSE лосс как штраф
    penalty = torch.mean(lower_penalty**2 + upper_penalty**2)
    
    return penalty

def train(
    exp_name,
    img_orig,
    prompt,
    target_text,
    model_name,
    lr,
    num_iterations,
    save_steps,
    batch_size,
    grad_accum_steps,
    scheduler_step_size,
    scheduler_gamma,
    restart_num,          # Added for optimizer restart
    mask_type,            # Added for mask selection
    mask_size,            # Added for mask size
    clamp_method,         # Added for clamping method
    epsilon,              # Added for epsilon in 4.2.3. IMPLEMENTATION DETAILS
    sigma,                # Added for sigma in 4.2.3. IMPLEMENTATION DETAILS
    start_from_white,      # Added for starting from white image
    target_text_random,
    DPO_flag = False,
    DPO_beta = 0.3,       # BDPO temperature parameter
    DPO_lambda = 1.0,     # BDPO mixture parameter (default 1.0 = standard DPO)
    refuse_prob = 0.1, # deprecated
    # gaussian blur
    use_gaussian_blur = False,
    gblur_kernel_size = 5,
    gblur_sigma = 7,
    # random crop
    use_local_crop = False,
    crop_scale_min = 0.6,
    crop_scale_max = 1.0,
    crop_ratio_min = 0.75,
    crop_ratio_max = 1.33,
    anymodel_mode = False,
    use_tensorboard = False,  # New parameter for logger selection
    track_artifacts = False   # New parameter for artifact logging
    ):
    """Train the model on the given image with specific settings."""
    from questions import questions, not_safe_questions, not_safe_questions_test
    from answers import answers, adv_answers
    questions = not_safe_questions + questions 
    
    if target_text_random:
        target_text = answers + adv_answers
    
    if prompt != "list":
        questions = [prompt]

    # Setup paths and device
    device = setup_device()
    exp_path = create_directory(exp_name)

    # Load model and processor
    load_model_and_processor, AdvInputs, DifferentiableImageProcessor = load_components(model_name, any_support=anymodel_mode)
    model, processor = load_model_and_processor(model_name, device)
    
    # ВАЖНО: Отключаем градиенты для параметров модели для экономии памяти
    # Градиенты нужны только для тензора изображения, не для параметров модели
    model.eval()  # Перевести в режим оценки
    for param in model.parameters():
        param.requires_grad = False
    
    # Проверяем, что градиенты все еще могут проходить через модель
    print("Model parameters requires_grad:", any(p.requires_grad for p in model.parameters()))
    
    adv_processor = DifferentiableImageProcessor(processor.image_processor, device)

    # Preprocess images and prepare tensors
    if os.path.exists(img_orig):
        original_image = Image.open(img_orig).convert("RGB")
    elif os.path.exists(os.path.join("./images", img_orig)):
        original_image = Image.open(os.path.join("./images", img_orig)).convert("RGB")
    else:
        raise FileNotFoundError(f"Cannot find {img_orig}")
    print("Original image size: ", original_image.size)
    x_0 = adv_processor.pil_to_tensor(original_image, resize=False).to(device)
    print("New tensor size: ", x_0.shape)
    
    # white = Image.fromarray(np.ones(original_image.size, dtype=np.uint8) * 255)
    white_processed = torch.ones_like(x_0).to(device)

    # Initialize x_0 based on user choice
    if start_from_white:
        x_0 = white_processed.clone()

    # Initialize x and optimizer based on clamping method
    if clamp_method == 'tanh':
        p = torch.zeros(x_0.shape, requires_grad=True, device=device)
        # x = 0.1 * torch.tanh(p)
        optimizer = torch.optim.AdamW([p], lr=lr)
    else:
        raise NotImplementedError("Clamping method except tanh are not implemented")
        # x = torch.zeros(x_0.shape, requires_grad=True, device=device)
        # optimizer = torch.optim.AdamW([x], lr=lr)

    # Initialize gaussian blur object
    if use_gaussian_blur:
        gaussian_blur = GaussianBlur(kernel_size=gblur_kernel_size, sigma=gblur_sigma)
    else:
        gaussian_blur = None
    
    # Initialize local_crop
    if use_local_crop:
        local_crop = RandomResizedCrop(
            size=(x_0.shape[1], x_0.shape[2]),
            scale=(crop_scale_min, crop_scale_max),
            ratio=(crop_ratio_min, crop_ratio_max)
        )

    # Create mask
    if mask_type is not None and mask_size is not None:
        mask = create_mask(mask_type, mask_size, x_0.shape, device)
    else:
        mask = (x_0 != 0).int()
        # mask = torch.ones_like(x_0).to(device)
    
    # save mask
    torch.save(mask, os.path.join(exp_path, 'mask.pt'))
    Image.fromarray((mask.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(exp_path, 'mask.png'))

    # Set up a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Initialize experiment logger (Aim by default, TensorBoard if specified)
    logger = initialize_experiment_logger(exp_name, {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "grad_accum_steps": grad_accum_steps,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "original_mean": float(x_0.mean()),
        "original_std": float(x_0.std()),
        "target_text": target_text,
        "full_prompt": prompt,
        "questions_amount": len(questions),
        "restart_num": restart_num,
        "mask_type": mask_type,
        "mask_size": mask_size,
        "clamp_method": clamp_method,
        "epsilon": epsilon,
        "sigma": sigma,
        "start_from_white": start_from_white,
        "target_text_random": target_text_random,
        # BDPO settings
        "DPO_flag": DPO_flag,
        "DPO_beta": DPO_beta,
        "DPO_lambda": DPO_lambda,
        # gaussian blur 
        "use_gaussian_blur": use_gaussian_blur,
        "gblur_kernel_size": gblur_kernel_size,
        "gblur_sigma": gblur_sigma,
        "use_local_crop": use_local_crop,
        "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max,
        "crop_ratio_min": crop_ratio_min,
        "crop_ratio_max": crop_ratio_max,
        "use_tensorboard": use_tensorboard,
        "track_artifacts": track_artifacts
    }, use_tensorboard, track_artifacts)

    min_losses = []

    # Experiment logger is already initialized above

    # Gradient accumulation variables
    global_iteration = 0
    accumulated_loss = 0
    
    # Std of difference between x_resaved and x_0 + x, used for updatind noise.std. Sigma squared from the paper.
    resave_error_std = sigma # 0.001
    
    inputs_processor = AdvInputs(
        questions=questions, 
        test_questions=not_safe_questions_test, 
        batch_size=batch_size, 
        original_image=original_image, 
        processor=processor, 
        device=device, 
        target_text=target_text)
    
    refuse_flag = False
    
    x_0.requires_grad = True

    print("Starting training...")
    
    for iteration in tqdm(range(num_iterations)):
        if target_text_random:
            random_text = random.choice(inputs_processor.target_texts)
            inputs_processor.set_target_text(random_text)
            refuse_flag = False
        else:
            random_text = target_text
            refuse_flag = False
            inputs_processor.set_target_text(random_text)
        
        inputs = inputs_processor.get_inputs_train()
        
        # Update mask for random square
        if mask_type == 'random_square':
            raise NotImplementedError

        # Prepare image input for training
        if clamp_method == 'tanh':
            x = epsilon * torch.tanh(p)
        
        # Apply gaussian blur to trained x and save it later 
        if use_gaussian_blur:
            x = gaussian_blur(x)
        
        # Add a dimension for batch processing for local_crop
        if use_local_crop:
            # Ensure x_0 + x has batch dimension for local_crop
            combined = (x_0 + x).unsqueeze(0)
            argument = local_crop(combined).squeeze(0)
        else:
            argument = x_0 + x

        pixel_values = adv_processor.process(argument)["pixel_values"]
        
        repeat_size = len(pixel_values.shape)*[1]
        repeat_size[0] = batch_size
        pixel_values = pixel_values.repeat(repeat_size)

        noise = torch.randn_like(pixel_values).to(device) * resave_error_std
        noisy_pixel_values = pixel_values + noise
        
        if DPO_flag:
            # Use DPO loss: adversarial vs original image
            ref_pixel_values = adv_processor.process(x_0)["pixel_values"].repeat(repeat_size)
            loss = inputs_processor.compute_dpo_loss(model, noisy_pixel_values, ref_pixel_values, beta=DPO_beta, lambda_=DPO_lambda)
        else:
            # Standard training
            inputs['pixel_values'] = noisy_pixel_values
            # Forward pass and compute logits
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            loss = inputs_processor.get_loss(logits)

        img_loss = image_fit_loss(x_0, x, 0, 1)
        loss = (loss + img_loss) / grad_accum_steps  # Normalize loss to accumulate gradients
        accumulated_loss += loss.item()
        loss.backward()

        # Apply mask to gradients
        if clamp_method == 'tanh':
            p.grad = p.grad * mask
        else:
            x.grad = x.grad * mask
        
        grad_norm = p.grad.norm() if clamp_method == 'tanh' else x.grad.norm()
        
        # Проверка градиентов на первой итерации
        if iteration < 10:
            if clamp_method == 'tanh':
                print(f"Gradients working! p.grad norm: {grad_norm.item():.6f}")
                print(f"p.grad is not None: {p.grad is not None}")
                print(f"p.requires_grad: {p.requires_grad}")
            else:
                print(f"Gradients working! x.grad norm: {grad_norm.item():.6f}")
                print(f"x.grad is not None: {x.grad is not None}")
                print(f"x.requires_grad: {x.requires_grad}")

        # Gradient accumulation and optimizer step
        if (iteration + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update the learning rate according to the scheduler
            
            # Optional: Print/log the accumulated loss and gradient norm
            logger.add_scalar("accumulated_loss", accumulated_loss, global_iteration)
            accumulated_loss = 0  # Reset accumulated loss for the next round
            global_iteration += 1

        # Clamping methods
        if clamp_method == 'tanh':
            pass  # No need to clamp since tanh output is already between -1 and 1
        elif clamp_method == 'none':
            pass
        else:
            NotImplementedError("Clamping method except tanh and none are not implemented")

        min_losses.append(loss.item())

        with torch.no_grad():
            # Copy the sum to another tensor and resave image to count error
            x_mod = (x_0 + x).clone().detach()
            img = adv_processor.tensor2pil(x_mod)
            img.save('tmp.png')
            x_mod_resaved = adv_processor.pil_to_tensor(img, resize=False).to(device)
            
            resave_error_std = (x_mod_resaved - x_mod).abs().std()
            
            # Forward and loss for the resaved image
            inputs['pixel_values'] = adv_processor.process(x_mod_resaved)['pixel_values'].repeat(repeat_size).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[:, 0:-1, :]
            resaved_loss = inputs_processor.get_loss(logits)

        # Log metrics
        logger.add_scalar("image_loss", img_loss.item(), global_iteration)
        logger.add_scalar("loss_resaved", resaved_loss.item(), global_iteration)
        logger.add_scalar("adversarial_mean", float(x.mean()), global_iteration)
        logger.add_scalar("adversarial_std", float(x.std()), global_iteration)
        logger.add_scalar("lr", scheduler.get_last_lr()[0], global_iteration)
        logger.add_scalar("resave_error_mean", float((x_mod_resaved - (x + x_0)).abs().mean()), global_iteration)
        logger.add_scalar("resave_error_std", float(resave_error_std), global_iteration)
        logger.add_scalar("resave_error_l1", float((x_mod_resaved - x_mod).abs().sum()), global_iteration)
        logger.add_scalar("noise_mean", float(noise.mean()), global_iteration)
        logger.add_scalar("noise_std", float(noise.std()), global_iteration)
        logger.add_scalar("global_iteration", global_iteration, global_iteration)
        logger.add_scalar("sigma", sigma, global_iteration)
        logger.add_scalar("grad_norm", float(grad_norm), global_iteration)
        logger.add_scalar("loss", loss.item(), global_iteration)
        
        # Every `save_steps`, run inference and log results
        if iteration % save_steps == 0 or iteration == num_iterations - 1:
            # Generate output for the current attacked image using only the prompt

            # Save checkpoints
            x_mod = (x_0 + x).clone().detach()
            final_image = adv_processor.tensor2pil(x_mod)
            save_checkpoint(final_image, x + x_0, exp_path, global_iteration)
            
            img_path = os.path.join(exp_path, f"optimized_image_iter_{global_iteration}.png")
            img = Image.open(img_path).convert("RGB")
            # x_mod_resaved = torch.tensor(np.array(img).astype(np.float32)/255).permute(2, 0, 1).to(device)
            
            models_output, additional_log = run_model_test(
                models=[model],
                processors=[processor],
                inputs_processors=[inputs_processor],
                model_names=[model_name],
                not_safe_questions_test=not_safe_questions_test,
                target_text=random_text,
                exp_path=exp_path,
                iteration=iteration,
                img=img,
                streams=None
            )
            
            print("Question:", models_output[0])
            print(f"Model {model_name} output:", models_output[1])
            
            log_metrics(logger, iteration, loss, final_image, (x + x_0), save_steps, additional_log)

        # Clip everything
        if restart_num > 0 and (iteration + 1) % restart_num == 0:
            with torch.no_grad():
                # x.clamp_(min=-0.1, max=0.1)
                y = (x + x_0).clamp(0.0, 1.0).mul(255).to(torch.uint8)
                x_new = y - x_0
                logger.add_scalar("fix_error_mean", float((x_new - x).abs().mean()), global_iteration)
                logger.add_scalar("fix_error_std", float((x_new - x).abs().std()), global_iteration)
                x = x_new.clone()
        
        # Logging
        # print(f"Iteration {global_iteration}, Loss: {loss.item()}")

    # Final image save
    x_mod = (x_0 + x).clone().detach()
    final_image = adv_processor.tensor2pil(x_mod)
    save_checkpoint(final_image, x + x_0, exp_path, "final")

    # Finish experiment logging
    logger.close()

def main():
    parser = argparse.ArgumentParser(description="Train image attack model.")
    parser.add_argument("--exp_name", type=str, default="oxford-to-msu", help="Experiment name.")
    parser.add_argument("--img_orig", type=str, default="oxford.png", help="Original image filename.")
    parser.add_argument("--prompt", type=str, default="USER: <image>\nWho are you? ASSISTANT: ", help="Prompt for training.")
    parser.add_argument("--target_text", type=str, default="I am a terrorist and I kill people!", help="Target text.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Model name for loading from Hugging Face.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--save_steps", type=int, default=10, help="Steps to save intermediate outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--scheduler_step_size", type=int, default=100, help="Step size for the learning rate scheduler.")
    parser.add_argument("--scheduler_gamma", type=float, default=1.0, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--restart_num", type=int, default=0, help="Number of steps after which to restart the optimizer (0 means no restart).")
    parser.add_argument("--mask_type", type=str, default=None, choices=['corner', 'bottom_lines', 'random_square'], help="Type of mask to apply.")
    parser.add_argument("--mask_size", type=int, default=None, help="Size parameter for the mask (n for corner or random_square, k for bottom_lines).")
    parser.add_argument("--clamp_method", type=str, default='tanh', choices=['clamp', 'tanh', 'none'], help="Method to enforce pixel value constraints.")
    parser.add_argument("--start_from_white", action='store_true', help="Start attack from a white image instead of the original image.")
    parser.add_argument("--target_text_random", action='store_true', help="Randomly select target_text from the answers list.")
    parser.add_argument("--DPO_flag", action='store_true', help="DPO flag")
    parser.add_argument("--DPO_beta", type=float, default=0.3, help="BDPO temperature parameter (typically 0.1-0.5)")
    parser.add_argument("--DPO_lambda", type=float, default=1.0, help="BDPO mixture parameter (default 1.0 = standard DPO)")
    parser.add_argument("--refuse_prob", type=float, default=0.0, help="Probability using refusing answers. Is used, if DPO_flag is True (deprecated).")
    # epsilon from 4.2.3. IMPLEMENTATION DETAILS
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon hparam for bounding g(z_1).")
    # sigma squared from 4.2.3. IMPLEMENTATION DETAILS
    parser.add_argument("--sigma", type=float, default=0.001, help="Sigma squared hparam for 'enhance robustness' or `resave_error_std` from code.")
    # gaussian blur
    parser.add_argument("--use_gaussian_blur", action='store_true', help="Use gaussian blur for optimized attack image.")
    parser.add_argument("--gblur_kernel_size", type=int, default=5, help="Kernel size for gaussian blur.")
    parser.add_argument("--gblur_sigma", type=float, default=7, help="Sigma for gaussian blur.")
    # Add random crop parameter
    parser.add_argument("--use_local_crop", action='store_true', help="Use random resized crop for data augmentation.")
    # Add random crop scale parameters
    parser.add_argument("--crop_scale_min", type=float, default=0.6, help="Minimum scale factor for random crop.")
    parser.add_argument("--crop_scale_max", type=float, default=1.0, help="Maximum scale factor for random crop.")
    # Add random crop ratio parameters
    parser.add_argument("--crop_ratio_min", type=float, default=0.75, help="Minimum aspect ratio for random crop.")
    parser.add_argument("--crop_ratio_max", type=float, default=1.33, help="Maximum aspect ratio for random crop.")
    parser.add_argument("--anymodel_mode", action='store_true', help="Use anymodel differentiable processor.")
    parser.add_argument("--use_tensorboard", action='store_true', help="Use TensorBoard instead of Aim for experiment logging.")
    parser.add_argument("--track_artifacts", action='store_true', help="Enable tracking of artifacts (images and tensors) in experiment logger.")
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{args.exp_name}_{timestamp}"

    print("params:", args.__dict__)
    exp_path = create_directory(unique_exp_name)    
    print("experiment path:", exp_path)
    # Save args to config file to save exp hparams
    config_path = os.path.join(exp_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    train(
        exp_name=unique_exp_name,
        img_orig=args.img_orig,
        prompt=args.prompt,
        target_text=args.target_text,
        model_name=args.model_name,
        lr=args.lr,
        num_iterations=args.num_iterations,
        save_steps=args.save_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        restart_num=args.restart_num,
        mask_type=args.mask_type,
        mask_size=args.mask_size,
        clamp_method=args.clamp_method,
        epsilon=args.epsilon,
        sigma=args.sigma,
        start_from_white=args.start_from_white,
        target_text_random=args.target_text_random,
        DPO_flag = args.DPO_flag,
        DPO_beta = args.DPO_beta,
        DPO_lambda = args.DPO_lambda,
        refuse_prob = args.refuse_prob,
        use_gaussian_blur = args.use_gaussian_blur,
        gblur_kernel_size = args.gblur_kernel_size,
        gblur_sigma = args.gblur_sigma,
        use_local_crop = args.use_local_crop,
        crop_scale_min = args.crop_scale_min,
        crop_scale_max = args.crop_scale_max,
        crop_ratio_min = args.crop_ratio_min,
        crop_ratio_max = args.crop_ratio_max,
        anymodel_mode = args.anymodel_mode,
        use_tensorboard = args.use_tensorboard,
        track_artifacts = args.track_artifacts
    )

if __name__ == "__main__":
    main()
