from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import argparse
import wandb  # Import WandB
import random  # Added import for random sampling
from torchvision.transforms import GaussianBlur # Additional regularization for noise

from processors import load_components
from train_test import run_model_test

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import GaussianBlur # Additional regularization for noise


def setup_device(i=0):
    """Setup computing device."""
    return torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")

def create_directory(exp_name, base_path="./runs"):
    """Creates a directory for experiment logs and outputs."""
    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def save_checkpoint(image: Image.Image, tensor: torch.Tensor, path: str, iteration: int):
    """Saves the current image checkpoint."""
    image.save(os.path.join(path, f"optimized_image_iter_{iteration}.png"))
    tensor.cpu().detach().numpy().astype(np.float32).tofile(os.path.join(path, f"optimized_image_iter_{iteration}.bin"))

def initialize_wandb(exp_name, config):
    """Initialize WandB for experiment tracking."""
    wandb.init(
        project="image_attack_optimization",
        name=exp_name,
        config=config,  # Logging configuration
        tags=["image-attack", "training", "transformers"],
        mode="online"
    )

def log_metrics_wandb(
        iteration: int,
        loss: torch.Tensor,
        final_image: np.array,
        final_tensor: torch.Tensor,
        generated_table: wandb.Table,
        save_steps: int
    ):
    """Logs metrics and images to WandB."""

    if iteration % save_steps == 0:  # Log images and model output every `save_steps` iterations
        wandb.log({"optimized_image": [wandb.Image(final_image, caption=f"Iteration {iteration}")]})
        # Log the generated text to WandB
        wandb.log({"generated_text": generated_table})
        wandb.log({"iteration": iteration})
        # log x+x_0
        wandb.log({"optimized_tensor": [wandb.Image(final_tensor, caption=f"Iteration {iteration}")]})

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
    # We compute the sum
    x_sum = (x_0 + x)
    
    lower_bound = torch.zeros(x_sum.shape).to(x_sum.device)
    upper_bound = torch.ones(x_sum.shape).to(x_sum.device)
    
    # Penalty for out-of-bounds
    lower_penalty = torch.relu(center_force*lower_bound - x_sum)  # if x_sum < center_force*lower_bound
    upper_penalty = torch.relu(x_sum - center_force*upper_bound)  # if x_sum > center_force*upper_bound
    
    # MSE-based penalty
    penalty = torch.mean(lower_penalty**2 + upper_penalty**2)
    
    return penalty

def pil_to_tensor(image: Image.Image, do_convert_rgb: bool = True, resize: bool = False) -> torch.Tensor:
    """
    Convert a PIL image to a tensor.

    Args:
        image: PIL image
        resize: Whether to resize the image to the optimal size for the model.

    Returns:
        A tensor image, shape (3, H, W)
    """
    if do_convert_rgb:
        image = image.convert("RGB")
    tensor_image = torch.tensor(np.array(image).astype(np.float32) / 255).permute(2, 0, 1)

    if resize:
        raise NotImplementedError("Resizing is not universal for models!!!")
    return tensor_image

def train(
    exp_name,
    img_orig,
    prompt,
    target_text,
    model_names,
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
    DPO_flag = True,
    refuse_prob = 0.1,
    model_weights = None,  # <-- NEW
    attack_norm = 0.5,
    # gaussian blur
    use_gaussian_blur = False,
    gblur_kernel_size = 5,
    # random crop
    use_local_crop = False,
    crop_scale_min = 0.6,
    crop_scale_max = 1.0,
    crop_ratio_min = 0.75,
    crop_ratio_max = 1.33
    ):
    """Train the model on the given image with specific settings."""
    from questions import questions, not_safe_questions, not_safe_questions_test
    from answers import answers, adv_answers
    questions = not_safe_questions + questions 

    refuse_flag = False
    
    if target_text_random:
        target_text = answers + adv_answers
    
    if prompt != "list":
        questions = [prompt]

    # Create exp directory
    exp_path = create_directory(exp_name)
    
    # Preprocess images and prepare tensors
    if os.path.exists(img_orig):
        original_image = Image.open(img_orig).convert("RGB")
    elif os.path.exists(os.path.join("./images", img_orig)):
        original_image = Image.open(os.path.join("./images", img_orig)).convert("RGB")
    else:
        raise FileNotFoundError(f"Cannot find {img_orig}")
    print("Original image size: ", original_image.size)

    # Initialize gaussian blur object
    if use_gaussian_blur:
        gaussian_blur = GaussianBlur(kernel_size=gblur_kernel_size)
    else:
        gaussian_blur = None

    # Load model and processor
    devices = []
    models = []
    processors = []
    adv_processors = []
    inputs_processors = []
    for i, model_name in enumerate(model_names):
        print(f"Loading {model_name}")
        print(f"Device number is {i}")
        devices.append(setup_device(i))
        load_model_and_processor, AdvInputs, DifferentiableImageProcessor = load_components(model_name)
        model, processor = load_model_and_processor(model_name, devices[-1])
        adv_processor = DifferentiableImageProcessor(processor.image_processor, devices[-1])
        inputs_processor = AdvInputs(
            questions=questions, 
            test_questions=not_safe_questions_test, 
            batch_size=batch_size, 
            original_image=original_image, 
            processor=processor, 
            device=devices[-1], 
            target_text=target_text)
        
        models.append(model)
        processors.append(processor)
        adv_processors.append(adv_processor)
        inputs_processors.append(inputs_processor)
    
    device_last = setup_device(len(devices) - 1)

    x_0 = pil_to_tensor(original_image, resize=False, do_convert_rgb=adv_processors[0].do_convert_rgb).to(device_last)
    # Initialize or override model_weights if needed
    if model_weights is None:
        # If not provided, default each model to weight 1.0
        model_weights = [1.0]*len(models)
    elif len(model_weights) != len(models):
        raise ValueError("The length of model_weights must match the number of model_names.")
    
    # Initialize x and optimizer based on clamping method
    if clamp_method == 'tanh':
        p = torch.zeros(x_0.shape, requires_grad=True, device=device_last)
        optimizer = torch.optim.AdamW([p], lr=lr)
    else:
        raise NotImplementedError("Clamping method except tanh are not implemented yet.")
    
    # Create mask
    if mask_type is not None and mask_size is not None:
        mask = create_mask(mask_type, mask_size, x_0.shape, device_last)
    else:
        mask = (x_0 != 0).int()

    # Initialize local_crop
    if use_local_crop:
        local_crop = RandomResizedCrop(
            size=(x_0.shape[1], x_0.shape[2]),
            scale=(crop_scale_min, crop_scale_max),
            ratio=(crop_ratio_min, crop_ratio_max)
        ) # DifferentiableRandomCrop

    # save mask
    torch.save(mask, os.path.join(exp_path, 'mask.pt'))
    Image.fromarray((mask.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(exp_path, 'mask.png'))

    # Set up a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Initialize WandB
    initialize_wandb(exp_name, {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "grad_accum_steps": grad_accum_steps,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "original_mean": x_0.mean(),
        "original_std": x_0.std(),
        "target_text": target_text,
        "full_prompt": prompt,
        "questions amount": len(questions),
        "restart_num": restart_num,
        "mask_type": mask_type,
        "mask_size": mask_size,
        "clamp_method": clamp_method,
        "epsilon": epsilon,
        "sigma": sigma,
        "start_from_white": start_from_white,
        "DPO_flag": DPO_flag,
        "refuse_prob": refuse_prob,
        "attack_norm": epsilon,
        "model_weights": model_weights,
        "use_gaussian_blur": use_gaussian_blur,
        "gblur_kernel_size": gblur_kernel_size,
        "use_local_crop": use_local_crop,
        "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max,
        "crop_ratio_min": crop_ratio_min,
        "crop_ratio_max": crop_ratio_max
    })

    min_losses = []

    # Создаём таблицу для логгирования выводов моделей
    model_outputs_table = wandb.Table(columns=["iteration"] + model_names)


    # Gradient accumulation variables
    global_iteration = 0
    accumulated_loss = 0
    
    # Std of difference between x_resaved and x_0 + x, used for updating noise.std
    resave_error_std = 0.001
    
    for iteration in tqdm(range(num_iterations)):
        # Possibly switch target_text if DPO_flag or random
        if DPO_flag or target_text_random:
            coin = random.random()
            print("coin:", coin)
            if DPO_flag and coin < refuse_prob:
                for inputs_processor in inputs_processors:
                    random_text = random.choice(inputs_processor.refuses)
                    inputs_processor.set_target_text(random_text)
                refuse_flag = True
            elif target_text_random:
                random_text = random.choice(inputs_processor.target_texts)
                for inputs_processor in inputs_processors:
                    inputs_processor.set_target_text(random_text)
                refuse_flag = False
            else:
                random_text = target_text
                for inputs_processor in inputs_processors:
                    inputs_processor.set_target_text(random_text)
                refuse_flag = False
            print(f"{refuse_flag} text:", random_text)
        
        # Update mask for random square (if needed)
        if mask_type == 'random_square':
            raise NotImplementedError("Dynamic random-square mask updating not implemented.")

        # Prepare image input for training
        if clamp_method == 'tanh':
            x = attack_norm * torch.tanh(p)

        inputss = []
        pgrads = []
        losses = []

        # Apply gaussian blur to trained x and save it later 
        if use_gaussian_blur:
            x =  gaussian_blur(x)
        
        argument = x_0 + x

        if use_local_crop:
            # Ensure x_0 + x has batch dimension for local_crop
            argument = local_crop(argument.unsqueeze(0)).squeeze(0)

        # Calculate penalty-based loss for staying in valid image range
        img_loss = image_fit_loss(x_0, x, 0, 1)

        # Zero out the gradient at the start of each iteration
        if p.grad is not None:
            p.grad.zero_()

        for i, inputs_processor in enumerate(inputs_processors):
            inputs = inputs_processor.get_inputs_train()
            inputss.append(inputs)

            pixel_values = adv_processors[i].process(argument.to(devices[i]))["pixel_values"]
            repeat_size = len(pixel_values.shape)*[1]
            repeat_size[0] = batch_size
            pixel_values = pixel_values.repeat(repeat_size)

            noise = torch.randn_like(pixel_values).to(device_last) * resave_error_std
            inputs['pixel_values'] = pixel_values + noise.to(devices[i])
            
            # Forward pass and compute logits
            outputs = models[i](**inputs)
            logits = outputs.logits[:, :-1, :]
            
            model_loss = inputs_processor.get_loss(logits)
            total_loss = model_weights[i] * model_loss + img_loss.to(devices[i]).clone()

            losses.append(total_loss.item())
            accumulated_loss += total_loss.item()

            if i < len(devices) - 1:
                total_loss.backward(retain_graph=True)
            else:
                total_loss.backward()

            # # gradient graph will be reset after image range loss
            # total_loss.backward(retain_graph=True)
            
            # Store gradient, then reset
            pgrads.append(p.grad.to(device_last).clone())
            p.grad.zero_()
            
        # img_loss.backward()
        # pgrads.append(p.grad.to(device_last).clone())
        # p.grad.zero_()
        
        # Sum up all model gradients in pgrads
        p.grad = torch.stack(pgrads).sum(dim=0)

        # Apply mask to gradient
        # Apply mask to gradients
        if clamp_method == 'tanh':
            p.grad = p.grad * mask
        else:
            x.grad = x.grad * mask
        
        grad_norm = p.grad.norm() if clamp_method == 'tanh' else x.grad.norm()

        # Gradient accumulation and optimizer step
        if (iteration + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update the learning rate according to the scheduler
            
            # Log the accumulated loss
            wandb.log({
                "accumulated_loss": accumulated_loss,
            })
            accumulated_loss = 0  # Reset accumulated loss for the next round
            global_iteration += 1

        # Clamping methods
        if clamp_method == 'tanh':
            pass  # No need to clamp since tanh output is already between -1 and 1
        elif clamp_method == 'none':
            pass
        else:
            NotImplementedError("Clamping method except tanh and none are not implemented")

        min_losses.append(sum(losses)/len(losses))

        with torch.no_grad():
            # Copy the sum to another tensor and re-save image to measure error
            x_mod = (x_0 + x).clone().detach().to(device_last)
            img = adv_processors[0].tensor2pil(x_mod)
            img.save('tmp.png')
            x_mod_resaved = pil_to_tensor(img, resize=False).to(device_last)
            
            resave_error_std = (x_mod_resaved - x_mod).abs().std()
            
            # Evaluate the re-saved image for logging
            resaved_losses = []
            for i, inputs in enumerate(inputss):
                pixel_values = adv_processors[i].process(x_mod_resaved.to(devices[i]))['pixel_values']
                repeat_size = len(pixel_values.shape)*[1]
                repeat_size[0] = batch_size
                inputs['pixel_values'] = pixel_values.repeat(repeat_size)
                outputs = models[i](**inputs)
                logits = outputs.logits[:, 0:-1, :]
                resaved_losses.append(inputs_processors[i].get_loss(logits))
            
            resaved_loss = torch.tensor([ls.item() for ls in resaved_losses]).mean()

        # Log metrics
        wandb_log_data = {
            "loss_resaved": resaved_loss.item(),
            "iteration": iteration,
            "adversarial_mean": x.mean(),
            "adversarial_std": x.std(),
            "lr": scheduler.get_last_lr()[0],
            "resave_error_mean": (x_mod_resaved - (x + x_0)).abs().mean(),
            "resave_error_std": resave_error_std,
            "resave_error_l1": (x_mod_resaved - x_mod).abs().sum(),
            "noise_mean": noise.mean(),
            "noise_std": noise.std(),
            "loss_per_iteration": sum(losses)/len(losses),
            "img_loss": img_loss.item(),  # <-- Log image penalty
            "global_iteration": global_iteration,
            "grad_norm": grad_norm,
            "use_gaussian_blur": use_gaussian_blur,
            "gblur_kernel_size": gblur_kernel_size
        }

        # Also log each model's partial loss
        for i, loss_val in enumerate(losses):
            mn = model_names[i].replace("/", "_")
            wandb_log_data[f"loss_{i}_{mn}"] = loss_val

        wandb.log(wandb_log_data)

        # Every `save_steps`, run inference and log results
        if iteration % save_steps == 0 or iteration == num_iterations - 1:
            x_mod = (x_0 + x).clone().detach()
            final_image = adv_processors[0].tensor2pil(x_mod)
            save_checkpoint(final_image, x + x_0, exp_path, global_iteration)
            
            img_path = os.path.join(exp_path, f"optimized_image_iter_{global_iteration}.png")
            img_for_test = Image.open(img_path).convert("RGB")
            
            iteration_outputs = [iteration]  # first column is iteration

            models_output, wandb_log = run_model_test(
                models=models,
                processors=processors,
                inputs_processors=inputs_processors,
                model_names=model_names,
                not_safe_questions_test=not_safe_questions_test,
                target_text="sure, here it is",
                exp_path=exp_path,
                iteration=iteration,
                img=img_for_test
            )
            
            wandb.log(wandb_log)

            iteration_outputs = iteration_outputs + models_output[1:]
            print("Question:", models_output[0])
            for i, ans in enumerate(models_output[1:]):
                print(f"Model {model_names[i]} output:", ans)
            
            model_outputs_table.add_data(*iteration_outputs)
            
            log_metrics_wandb(iteration, sum(losses)/len(losses), final_image, (x + x_0), model_outputs_table, save_steps)

        # If configured, clamp/clip every restart_num steps
        if restart_num > 0 and (iteration + 1) % restart_num == 0:
            with torch.no_grad():
                y = (x + x_0).clamp(0.0, 1.0).mul(255).to(torch.uint8)
                x_new = y - x_0
                wandb.log({
                    "fix_error_mean": (x_new - x).abs().mean(),
                    "fix_error_std": (x_new - x).abs().std()
                })
                x = x_new.clone()
        
    # Final image save
    x_mod = (x_0 + x).clone().detach()
    final_image = adv_processors[0].tensor2pil(x_mod)
    save_checkpoint(final_image, x + x_0, exp_path, "final")

    wandb.finish()


def parse_model_names(model_names_str):
    return model_names_str.split(',')


def main():
    parser = argparse.ArgumentParser(description="Train image attack model.")
    parser.add_argument("--exp_name", type=str, default="oxford-to-msu", help="Experiment name.")
    parser.add_argument("--img_orig", type=str, default="oxford.png", help="Original image filename.")
    parser.add_argument("--prompt", type=str, default="USER: <image>\nWho are you? ASSISTANT: ", help="Prompt for training.")
    parser.add_argument("--target_text", type=str, default="I am a terrorist and I kill people!", help="Target text.")
    parser.add_argument("--model_names", type=parse_model_names, default=["llava-hf/llava-1.5-7b-hf"], help="Model name for loading from Hugging Face.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--save_steps", type=int, default=10, help="Steps to save intermediate outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--scheduler_step_size", type=int, default=100, help="Step size for the learning rate scheduler.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.9, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--restart_num", type=int, default=0, help="Number of steps after which to restart the optimizer (0 means no restart).")  # Added argument
    parser.add_argument("--mask_type", type=str, default=None, choices=['corner', 'bottom_lines', 'random_square'], help="Type of mask to apply.")  # Added argument
    parser.add_argument("--mask_size", type=int, default=None, help="Size parameter for the mask (n for corner or random_square, k for bottom_lines).")  # Added argument
    parser.add_argument("--clamp_method", type=str, default='tanh', choices=['clamp', 'tanh', 'none'], help="Method to enforce pixel value constraints.")  # Updated default
    parser.add_argument("--start_from_white", action='store_true', help="Start attack from a white image instead of the original image.")  # Added argument
    parser.add_argument("--target_text_random", action='store_true', help="Randomly select target_text from the answers list.")
    parser.add_argument("--DPO_flag", action='store_true', help="DPO flag")
    parser.add_argument("--refuse_prob", type=float, default=0.0, help="Probability of using refusing answers. Used if DPO_flag is True.")
    parser.add_argument("--epsilon", type=float, default=0.4, help="Decrease to make attack more imperceptable. Values from 0 (no attack) to 0.5-0.6 (may lead to increased img resaving loss).")
    # sigma squared from 4.2.3. IMPLEMENTATION DETAILS
    parser.add_argument("--sigma", type=float, default=0.001, help="Sigma squared hparam for 'enhance robustness' or `resave_error_std` from code.")
    # gaussian blur
    parser.add_argument("--use_gaussian_blur", action='store_true', help="Use gaussian blur for training.")
    parser.add_argument("--gblur_kernel_size", type=int, default=5, help="Kernel size for gaussian blur.")
    # Add random crop parameter
    parser.add_argument("--use_local_crop", action='store_true', help="Use random resized crop for data augmentation.")
    # Add random crop scale parameters
    parser.add_argument("--crop_scale_min", type=float, default=0.6, help="Minimum scale factor for random crop.")
    parser.add_argument("--crop_scale_max", type=float, default=1.0, help="Maximum scale factor for random crop.")
    # Add random crop ratio parameters
    parser.add_argument("--crop_ratio_min", type=float, default=0.75, help="Minimum aspect ratio for random crop.")
    parser.add_argument("--crop_ratio_max", type=float, default=1.33, help="Maximum aspect ratio for random crop.")
    
    # NEW ARGUMENT FOR MODEL WEIGHTS
    parser.add_argument(
        "--model_weights",
        type=float,
        nargs='+',
        default=None,
        help="Loss weights for each model, must match length of model_names (default is 1.0 for each)."
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{args.exp_name}_{timestamp}"

    train(
        exp_name=unique_exp_name,
        img_orig=args.img_orig,
        prompt=args.prompt,
        target_text=args.target_text,
        model_names=args.model_names,
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
        DPO_flag=args.DPO_flag,
        refuse_prob=args.refuse_prob,
        model_weights=args.model_weights,
        use_gaussian_blur=args.use_gaussian_blur,
        gblur_kernel_size=args.gblur_kernel_size,
        use_local_crop=args.use_local_crop,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        crop_ratio_min=args.crop_ratio_min,
        crop_ratio_max=args.crop_ratio_max
    )

if __name__ == "__main__":
    main()
