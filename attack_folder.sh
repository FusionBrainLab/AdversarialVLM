#!/bin/bash

# File containing the WandB API key
WANDB_KEY_FILE="wandb_key.txt"

# Check if the WandB key file exists
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Error: WandB key file '$WANDB_KEY_FILE' not found."
    exit 1
fi

# Read the WandB API key from the file
WANDB_KEY=$(<"$WANDB_KEY_FILE")

# Log in to WandB using the API key
wandb login "$WANDB_KEY"

# Define all required arguments for the Python script
EXP_NAME="masked_folder_attack"
IMG_FOLDER="1000_images"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
LR=1e-2
NUM_ITERATIONS=50000
SAVE_STEPS=500
BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # WILL NOT WORK FOR FOLDER ATTACK

# Run the Python script with all arguments
wandb online
CUDA_VISIBLE_DEVICES=7 python attack_list_folder.py \
    --exp_name "$EXP_NAME" \
    --img_folder "$IMG_FOLDER" \
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
    --clamp_method "$CLAMP_METHOD"