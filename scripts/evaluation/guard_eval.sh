#!/bin/bash

# Script for guard evaluation using Gemma
# Usage: ./scripts/evaluation/guard_eval.sh RESULTS_PATH CUDA_NUM

if [ $# -ne 2 ]; then
    echo "Usage: $0 RESULTS_PATH CUDA_NUM"
    echo "Example: $0 /path/to/safebench/results 0"
    exit 1
fi

RESULTS_PATH=$1
CUDA_NUM=$2

echo "Running guard evaluation for results in: $RESULTS_PATH"

# Check if results path exists
if [ ! -d "$RESULTS_PATH" ]; then
    echo "Error: Results path $RESULTS_PATH does not exist"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_NUM

# Run guard evaluation
python src/evaluation/guard_eval_gemma.py "$RESULTS_PATH" "$CUDA_NUM"

echo "Guard evaluation completed. Check $RESULTS_PATH for results_gemma.csv and mean_result_gemma.txt" 