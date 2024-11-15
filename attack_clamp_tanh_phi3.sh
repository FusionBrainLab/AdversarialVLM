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
EXP_NAME="airi_tanh_Phi-3.5-vision_start"
IMG_ORIG="airi.png"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
LR=1e-2
NUM_ITERATIONS=5000
SAVE_STEPS=50
BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations

# Run the Python script with all arguments
wandb online
CUDA_VISIBLE_DEVICES=4 python "attack_list_o1upd_phi3.py" \
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
    --restart_num "$RESTART_NUM"
wait