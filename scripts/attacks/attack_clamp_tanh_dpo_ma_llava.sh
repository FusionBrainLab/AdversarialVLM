#!/bin/bash
source activate /home/jovyan/.mlspace/envs/rah_11_cu12.4_torch/
which conda
echo $CONDA_PREFIX



# Define all required arguments for the Python script
EXP_NAME="gray_dpo_attack_LlaVA-1.5-7B_bdpo_b03_l03-ma"
IMG_ORIG="gray.png"
PROMPT="list"
TARGET_TEXT="sure, here it is!"
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
LR=1e-2
NUM_ITERATIONS=3000  # Fewer iterations for DPO as it converges faster
SAVE_STEPS=50
BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
CLAMP_METHOD="tanh"  # Use tanh clamping method choices=['clamp', 'tanh', 'none']
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations
DPO_BETA=0.3  # DPO temperature parameter
DPO_lambda=0.3

CUDA_VISIBLE_DEVICES=4 python "src/attack_model.py" \
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
    --DPO_flag \
    --DPO_beta "$DPO_BETA" \
    --DPO_lambda "$DPO_lambda" \
    --target_text_random 
    # --use_local_crop

wait 