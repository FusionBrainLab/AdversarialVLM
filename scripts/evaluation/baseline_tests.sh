#!/bin/bash

# Script for baseline testing (MM-SafetyBench and FigStep)
# Usage: ./scripts/evaluation/baseline_tests.sh TEST_TYPE MODEL_SUFFIX CUDA_NUM [IMAGE_TYPE]

if [ $# -lt 3 ]; then
    echo "Usage: $0 TEST_TYPE MODEL_SUFFIX CUDA_NUM [IMAGE_TYPE]"
    echo "TEST_TYPE: mmsafety or figstep"
    echo "MODEL_SUFFIX: phi35, qwenVL, Llama32, llava-hf"
    echo "IMAGE_TYPE (for mmsafety): SD, TYPO, SD_TYPO (default: SD_TYPO)"
    echo ""
    echo "Examples:"
    echo "  $0 mmsafety phi35 0 SD_TYPO"
    echo "  $0 figstep Llama32 1"
    exit 1
fi

TEST_TYPE=$1
MODEL_SUF=$2
CUDA_NUM=$3
IMAGE_TYPE=${4:-"SD_TYPO"}

echo "Running $TEST_TYPE baseline test with model: $MODEL_SUF on GPU: $CUDA_NUM"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_NUM

if [ "$TEST_TYPE" = "mmsafety" ]; then
    echo "Running MM-SafetyBench baseline with image type: $IMAGE_TYPE"
    python src/evaluation/MM_SafetyBench_baseline.py \
        --model_suf "$MODEL_SUF" \
        --cuda_num "$CUDA_NUM" \
        --image_type "$IMAGE_TYPE"
elif [ "$TEST_TYPE" = "figstep" ]; then
    echo "Running FigStep baseline"
    python src/evaluation/FigStep_baseline.py \
        --model_suf "$MODEL_SUF" \
        --cuda_num "$CUDA_NUM"
else
    echo "Error: Unknown test type $TEST_TYPE. Use 'mmsafety' or 'figstep'"
    exit 1
fi

echo "Baseline testing completed." 