#!/bin/bash

# Define all required arguments for the Python script
EXP_NAME="gray_Qwen2-VL-2B_bdpo_b03_l03"
IMG_ORIG="gray.png"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
LR=1e-2
NUM_ITERATIONS=5000
SAVE_STEPS=50
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations
DPO_BETA=0.3  # DPO temperature parameter
DPO_lambda=0.3

# Run the Python script with all arguments
CUDA_VISIBLE_DEVICES=0 python "src/attack_model.py" \
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
    --restart_num "$RESTART_NUM"   \
    --DPO_flag \
    --DPO_beta "$DPO_BETA" \
    --DPO_lambda "$DPO_lambda"
wait