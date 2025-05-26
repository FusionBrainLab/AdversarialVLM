#!/bin/bash

# Script for finding best iteration using Gemma judge
# Usage: ./scripts/evaluation/find_best_iter.sh

echo "Starting best iteration search with Gemma judge..."

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Run the best iteration finder
python src/evaluation/find_best_iter_gemma.py

echo "Best iteration search completed. Check runs/ directories for results." 