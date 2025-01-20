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

from processors import load_components
from train_test import run_model_test

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
    start_from_white,      # Added for starting from white image
    target_text_random
    ):
    """Train the model on the given image with specific settings."""
    from questions import questions, not_safe_questions, not_safe_questions_test
    from answers import answers, adv_answers
    questions = not_safe_questions + questions 
    
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
    
    device_last = setup_device(3)

    x_0 = pil_to_tensor(original_image, resize=False, do_convert_rgb=adv_processors[0].do_convert_rgb).to(device_last)
    # print("New tensor size: ", x_0.shape)

    # Initialize x and optimizer based on clamping method
    if clamp_method == 'tanh':
        p = torch.zeros(x_0.shape, requires_grad=True, device=device_last)
        # x = 0.1 * torch.tanh(p)
        optimizer = torch.optim.AdamW([p], lr=lr)
    else:
        raise NotImplementedError("Clamping method except tanh are not implemented")
        # x = torch.zeros(x_0.shape, requires_grad=True, device=device)
        # optimizer = torch.optim.AdamW([x], lr=lr)
    
    # Create mask
    if mask_type is not None and mask_size is not None:
        mask = create_mask(mask_type, mask_size, x_0.shape, device_last)
    else:
        mask = (x_0 != 0).int()
        # mask = torch.ones_like(x_0).to(device)

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
        "start_from_white": start_from_white
    })

    min_losses = []

    # Создаём таблицу для логгирования выводов моделей
    model_outputs_table = wandb.Table(columns=["iteration"] + model_names)


    # Gradient accumulation variables
    global_iteration = 0
    accumulated_loss = 0
    
    # Std of difference between x_resaved and x_0 + x, used for updatind noise.std
    resave_error_std = 0.001
    
    # Compute target tokens ones

    for iteration in tqdm(range(num_iterations)):
        if target_text_random:
            random_text = random.choice(inputs_processor.target_texts)
            for inputs_processor in inputs_processors:
                inputs_processor.set_target_text(random_text)
        # Update mask for random square
        if mask_type == 'random_square':
            raise NotImplementedError
        # Prepare image input for training
        if clamp_method == 'tanh':
            x = 0.5 * torch.tanh(p)
        
        inputss = []
        pgrads = []
        losses = []
        for i, inputs_processor in enumerate(inputs_processors):
            inputs = inputs_processor.get_inputs_train()
            inputss.append(inputs)
             
            pixel_values = adv_processors[i].process((x_0+x).to(devices[i]))["pixel_values"]
        
            repeat_size = len(pixel_values.shape)*[1]
            repeat_size[0] = batch_size
            pixel_values = pixel_values.repeat(repeat_size)
            noise = torch.randn_like(pixel_values).to(device_last) * resave_error_std
            inputs['pixel_values'] = pixel_values + noise.to(devices[i])
            
            # Forward pass and compute logits
            outputs = models[i](**inputs)
            logits = outputs.logits[:, :-1, :]
            
            loss = inputs_processor.get_loss(logits)
            losses.append(loss.item())
            accumulated_loss += loss.item()
            if i < len(devices) - 1:
                loss.backward(retain_graph=True)    
            else:
                loss.backward()
            print(p.grad.norm())
            pgrads.append(p.grad.to(device_last))

            print(f"Loss for model {i}: ", loss)
 
        p.grad = torch.stack(pgrads).sum(dim=0)
        
        # img_loss = image_fit_loss(x_0, x, 0, 1)
        # loss = (loss + img_loss) / grad_accum_steps  # Normalize loss to accumulate gradients
        # accumulated_loss += loss.item()
        # loss.backward()

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
            
            # Optional: Print/log the accumulated loss and gradient norm
            # print(f"Step {global_iteration}, Accumulated Loss: {accumulated_loss}, Grad Norm: {grad_norm.item()}")
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

        min_losses.append(loss.item())

        with torch.no_grad():
            # Copy the sum to another tensor and resave image to count error
            x_mod = (x_0 + x).clone().detach().to(device_last)
            img = adv_processors[0].tensor2pil(x_mod)
            img.save('tmp.png')
            x_mod_resaved = pil_to_tensor(img, resize=False).to(device_last)
            
            resave_error_std = (x_mod_resaved - x_mod).abs().std()
            
            # Forward and loss for the resaved image
            resaved_losses = []
            for i, inputs in enumerate(inputss):
                pixel_values = adv_processors[i].process(x_mod_resaved.to(devices[i]))['pixel_values']
                repeat_size = len(pixel_values.shape)*[1]
                repeat_size[0] = batch_size
                inputs['pixel_values'] = pixel_values.repeat(repeat_size)
                outputs = models[i](**inputs)
                logits = outputs.logits[:, 0:-1, :]
                resaved_losses.append(inputs_processors[i].get_loss(logits))
            
            resaved_loss = torch.tensor([loss.item() for loss in resaved_losses]).mean()
        
        # Log metrics dynamically based on the number of models
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
            "loss": loss.item(),
            "global_iteration": global_iteration,
            "grad norm": grad_norm
        }

        # Add losses for each model dynamically
        for i, model_loss in enumerate(losses):
            mn = model_names[i].replace("/", "_")
            wandb_log_data[f"loss_{i}_{mn}"] = model_loss

        # Log all collected data to WandB
        wandb.log(wandb_log_data)

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
            
            iteration_outputs = [iteration]  # Первая колонка — номер итерации
    
            models_output, wandb_log = run_model_test(
                models=models,
                processors=processors,
                inputs_processors=inputs_processors,
                model_names=model_names,
                not_safe_questions_test=not_safe_questions_test,
                target_text="sure, here it is",
                exp_path=exp_path,
                iteration=iteration,
                img=img
            )
            
            wandb.log(wandb_log)

            # Добавляем строку в таблицу
            iteration_outputs = iteration_outputs + models_output[1:]
            
            print("Question:", models_output[0])
            for ans in models_output[1:]:
                print(f"Model {model_names[i]} output:", ans)
            
            model_outputs_table.add_data(*iteration_outputs)
            
            log_metrics_wandb(iteration, loss, final_image, (x + x_0), model_outputs_table, save_steps)

        # Clip everything
        if restart_num > 0 and (iteration + 1) % restart_num == 0:
            with torch.no_grad():
                # x.clamp_(min=-0.1, max=0.1)
                y = (x + x_0).clamp(0.0, 1.0).mul(255).to(torch.uint8)
                x_new = y - x_0
                wandb.log({
                    "fix_error_mean": (x_new - x).abs().mean(),
                    "fix_error_std": (x_new - x).abs().std()
                })
                x = x_new.clone()
        
        # Logging
        # print(f"Iteration {global_iteration}, Loss: {loss.item()}")

    # Final image save
    final_image = adv_processor.tensor2pil(final_image)
    save_checkpoint(final_image, x + x_0, exp_path, "final")

    # Finish WandB run
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
    parser.add_argument("--clamp_method", type=str, default='clamp', choices=['clamp', 'tanh', 'none'], help="Method to enforce pixel value constraints.")  # Added argument
    parser.add_argument("--start_from_white", action='store_true', help="Start attack from a white image instead of the original image.")  # Added argument
    parser.add_argument("--target_text_random", action='store_true', help="Randomly select target_text from the answers list.")
    
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
        restart_num=args.restart_num,              # Passed new argument
        mask_type=args.mask_type,                  # Passed new argument
        mask_size=args.mask_size,                  # Passed new argument
        clamp_method=args.clamp_method,            # Passed new argument
        start_from_white=args.start_from_white,
        target_text_random=args.target_text_random
    )

if __name__ == "__main__":
    main()
