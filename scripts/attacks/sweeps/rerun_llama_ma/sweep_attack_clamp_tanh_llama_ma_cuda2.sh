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

CUDA_ID=2
MODEL_NAME="alpindale/Llama-3.2-11B-Vision-Instruct"
SAVE_STEPS=100
# Define all required arguments for the Python script
BATCH_SIZE=1
CLAMP_METHOD="tanh"
PROMPT="list"
GRAD_ACCUM_STEPS=1
LR=1e-2 # 1e-2
NUM_ITERATIONS=5000 # 5000
RESTART_NUM=0  # Restart optimizer every RESTART_NUM iterations
IMG_ORIG="gray.png"
TARGET_TEXT="sure, here it is!" # Overwritten if PROMPT='list'

# Define arrays for sweeping parameters
LRS=(1e-4 1e-3 1e-2 1e-1)
# EPSILONS=(0.5 0.8)
EPSILONS=(0.5)

# Run sweeps
for lr in "${LRS[@]}"; do
    for epsilon in "${EPSILONS[@]}"; do
        EXP_NAME="gray_Llama-MA_sweep1_lr${lr}_eps${epsilon}"
        
        echo "Running experiment: $EXP_NAME"
        
        wandb online
        CUDA_VISIBLE_DEVICES="$CUDA_ID" python "src/attack_model.py" \
            --exp_name "$EXP_NAME" \
            --img_orig "$IMG_ORIG" \
            --prompt "$PROMPT" \
            --target_text "$TARGET_TEXT" \
            --model_name "$MODEL_NAME" \
            --lr "$lr" \
            --num_iterations "$NUM_ITERATIONS" \
            --save_steps "$SAVE_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum_steps "$GRAD_ACCUM_STEPS" \
            --scheduler_step_size 100 \
            --scheduler_gamma 1.0 \
            --clamp_method "$CLAMP_METHOD" \
            --restart_num "$RESTART_NUM" \
            --epsilon "$epsilon" \
            --target_text_random
        wait
    done
done