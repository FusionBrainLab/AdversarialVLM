import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
project_root = os.path.abspath("/home/jovyan/rahmatullaev/adversarial/src")
sys.path.append(project_root)

from src.processors import load_components
from src.judge.safety_checker import SafetyChecker

def main(SB_ans_path, cuda_num=0):
    model_name = "google/gemma-3-4b-it"
    device = f"cuda:{cuda_num}"
    dtype = 'bfloat16'
    checker = SafetyChecker(model_name, device, dtype)

    asr_results = []
    guard_ans = []

    # Iterate over 23 categories
    for i in tqdm(range(1, 24), leave=True):
        # Load data for both attacked and reference tables
        sfb_data_i = pd.read_csv(os.path.join(SB_ans_path, f"{i}.csv"), names=[0])
        guard_asr_i = []

        for line_idx in tqdm(range(len(sfb_data_i)), leave=False):
            # Process inputs in batches
            inputs = [text for text in sfb_data_i.iloc[line_idx]]

            with torch.no_grad(): 
                output = checker.evaluate_text(inputs, max_tokens=10)
            guard_asr_i.append(output)

        guard_ans.append(guard_asr_i)
        
        # Count unsafe classifications
        attack_unsafe = sum((not response[0].is_safe) for response in guard_asr_i) / len(guard_asr_i)

        asr_results.append([attack_unsafe])

        print(f"Set {i}: ASR = {attack_unsafe}")

    # Save results
    results_df = pd.DataFrame(asr_results, columns=["asr"])
    results_df.to_csv(os.path.join(SB_ans_path, "results_gemma.csv"), index=False)
    
    mean = results_df["asr"].mean()
    with open(os.path.join(SB_ans_path, "mean_result_gemma.txt"), "w") as f:
        f.write(str(mean))

    guard_ans = pd.DataFrame(guard_ans, index=list(range(1,24))).T
    guard_ans.to_csv(os.path.join(SB_ans_path, "guard_gemma.csv"), index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial Model Testing")
    parser.add_argument("path", type=str, help="Experiment SB results")
    parser.add_argument("cuda_num", type=int, default=0, help="Cuda number")

    args = parser.parse_args()
    main(args.path, args.cuda_num)
