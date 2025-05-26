#!/bin/bash
source activate /home/jovyan/.mlspace/envs/rah_11_cu12.4_torch/
which conda
echo $CONDA_PREFIX

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
EXP_NAME="gray_Llama_localization_eps0.4_smaller_crop_init"
IMG_ORIG="Llama_optimized_image_iter_3501.png"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="alpindale/Llama-3.2-11B-Vision-Instruct" #  "SinclairSchneider/Llama-Guard-3-11B-Vision"
LR=1e-2
NUM_ITERATIONS=50000
SAVE_STEPS=250
BATCH_SIZE=2
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations

# Run the Python script with all arguments
wandb online
CUDA_VISIBLE_DEVICES=0 python "src/attack_model_M-fork.py" \
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
    --epsilon 0.4 \
    --use_local_crop \
    --crop_scale_min 0.9 \
    --crop_scale_max 1.0 \
    --crop_ratio_min 0.9 \
    --crop_ratio_max 1.1 \
     #--target_text_random
wait