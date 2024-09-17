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
EXP_NAME="oxford-to-sure-2"
IMG_ORIG="optimized_image_iter_final.png"
IMG_REF="msu.png"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="llava-hf/llava-1.5-7b-hf" # "openbmb/MiniCPM-V-2"
LR=1e-2
NUM_ITERATIONS=100
SAVE_STEPS=10
BATCH_SIZE=4
GRAD_ACCUM_STEPS=1 

# Run the Python script with all arguments
wandb online
python attack_list.py \
    --exp_name "$EXP_NAME" \
    --img_orig "$IMG_ORIG" \
    --img_ref "$IMG_REF" \
    --prompt "$PROMPT" \
    --target_text "$TARGET_TEXT" \
    --model_name "$MODEL_NAME" \
    --lr "$LR" \
    --num_iterations "$NUM_ITERATIONS" \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --scheduler_step_size 100 \
    --scheduler_gamma 1.0
