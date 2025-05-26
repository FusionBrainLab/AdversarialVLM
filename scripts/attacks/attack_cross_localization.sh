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
# wandb login "$WANDB_KEY"

# Define all required arguments for the Python script
EXP_NAME="gray_crossattack_phi_llama_qwen_llava"
IMG_ORIG="gray.png"
PROMPT="list" # "list" for all questions in the list
TARGET_TEXT="sure, here it is"
MODEL_NAME="microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf" 
# "alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf" 
# "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf" 
# "microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct" 
# "microsoft/Phi-3.5-vision-instruct,alpindale/Llama-3.2-11B-Vision-Instruct,Qwen/Qwen2-VL-2B-Instruct,llava-hf/llava-1.5-7b-hf" 
LR=1e-2
ATTACK_NORM=0.4
NUM_ITERATIONS=20000
SAVE_STEPS=100
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations

# Run the Python script with all arguments
wandb online
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "src/crossattack_models_M-fork.py" \
    --exp_name "$EXP_NAME" \
    --img_orig "$IMG_ORIG" \
    --prompt "$PROMPT" \
    --target_text "$TARGET_TEXT" \
    --model_name "$MODEL_NAME" \
    --lr "$LR" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --scheduler_step_size 100 \
    --scheduler_gamma 1.0 \
    --clamp_method "$CLAMP_METHOD" \
    --restart_num "$RESTART_NUM" \
    --epsilon "$ATTACK_NORM" # --target_text_random
    # --use_local_crop \
    # --crop_scale_min 0.9 \
    # --crop_scale_max 1.0 \
    # --crop_ratio_min 0.9 \
    # --crop_ratio_max 1.1   

wait