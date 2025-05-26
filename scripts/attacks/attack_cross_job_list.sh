#!/bin/bash

# File containing the WandB API key
WANDB_KEY_FILE="wandb_key.txt"

# Check if the WandB key file exists
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Error: WandB key file '$WANDB_KEY_FILE' not found."
    exit 1
fi

# Read the WandB API key from the file
export WANDB_API_KEY=$(cat <"$WANDB_KEY_FILE")
WANDB_KEY=$(<"$WANDB_KEY_FILE")
echo $WANDB_KEY

# Log in to WandB using the API key
wandb online

# Function to run the training script with specific parameters
run_training() {
    local EXP_NAME="$1"
    local MODEL_NAME="$2"

    python "src/crossattack_models.py" \
        --exp_name "$EXP_NAME" \
        --img_orig "gray.png" \
        --prompt "list" \
        --target_text "sure, here it is" \
        --model_name "$MODEL_NAME" \
        --lr 1e-2 \
        --num_iterations 20000 \
        --save_steps 250 \
        --batch_size 1 \
        --grad_accum_steps 1 \
        --scheduler_step_size 100 \
        --scheduler_gamma 1.0 \
        --clamp_method "tanh" \
        --restart_num 0 \
        --target_text_random \
        --use_local_crop
}

# Run the trainings sequentially
run_training "gray_crossattack_llama_qwen_llava_MA_localization" "alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf"
wait

run_training "gray_crossattack_phi_qwen_llava_MA_localization" "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf"
wait

run_training "gray_crossattack_phi_llama_llava_MA_localization" "microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,llava-hf/llava-1.5-7b-hf"
wait

run_training "gray_crossattack_phi_qwen_llama_MA_localization" "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-2B-Instruct,alpindale/Llama-3.2-11B-Vision-Instruct"
wait
