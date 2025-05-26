import os
import sys

project_root = os.path.abspath("/home/jovyan/rahmatullaev/adversarial/")
sys.path.append(project_root)

import torch
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from src.processors import load_components
from src.judge.safety_checker import SafetyChecker


#########################################################
# Configuration
#########################################################


exps = [
    # "gray_Phi-3.5_20250121_110223",
    # "gray_Phi-3.5-MA-vs-test_20250116_182955",
    # "gray_Phi-3.5_localization_20250513_204659",
    # "gray_Phi-3.5_gblurON_ksize5_MA_savex_20250514_191757",
    # "gray_Llama_20250121_110131",
    # "gray_Llama-MA-vs-test_20250116_195545",
    # "gray_Qwen2-VL-2B_20250121_191132",
    # "gray_Qwen2-VL-2B-MA-vs-test_20250116_190331",
    # "gray_Qwen2-VL-2B_localization_20250513_221642",
    # "gray_Qwen2-VL-2B_gblurON_ksize5_MA_savepixel_20250513_202458",
    # "gray_LlaVA-1.5-7B_20250121_185904",
    # "gray_LlaVA-1.5-7B-MA_20250121_185952",
    # "gray_LlaVA-1.5-7B_localization_20250513_204713",
    # "gray_crossattack_phi_llama_qwen_0.4_20250206_201536",
    # "gray_crossattack_phi_qwen_llava_20250123_011655",
    # "istanbulHD_crossattack_phi_llama_llava_0.1_20250130_013257",
    # "gray_crossattack_phi3_llama_qwen_20241219_213034",
    # "gray_crossattack_phi3_llama_qwen_20241219_213034_copy",
    # "gray_crossattack_llama_qwen_llava_20250123_011738",
    # "gray_crossattack_phi_llama_llava_20250123_175253",
    # "gray_crossattack_llama_qwen_llava_MA_20250123_205135",
    # "gray_crossattack_phi_qwen_llava_MA_20250124_040413",
    # "gray_crossattack_phi_llama_llava_MA_20250124_080934",
    # "gray_crossattack_phi3_llama_qwen-MA_20250117_010701",
    # "gray_Qwen2-VL-2B_gblurON_DEBUG_20250508_201302",
    # "gray_Qwen2-VL-2B_gblurOFF_DEBUG_20250510_025340",
    # "gray_Qwen2-VL-2B_gblurON_DEBUG_20250510_190832",
    # "gray_Qwen2-VL-2B_gblurOFF_DEBUG_20250510_171227",
    # "gray_Qwen2-VL-2B_gblurON_ksize5_MA_savex_20250513_175947",
    # "gray_Qwen2-VL-2B_gblurON_ksize5_MA_savepixel_20250513_202458",
    # "gray_Qwen2-VL-2B_localization_20250513_221642",
    # "gray_Phi-3.5_localization_20250513_204659",
    # "gray_Phi-3.5_gblurON_ksize5_MA_savex_20250514_191757",
    # "gray_Llama_gblurON_ksize5_MA_savex_20250514_203529",
    # "gray_Llama_gblurON_ksize5_savex_20250515_013348",
    # "gray_Qwen2-VL-2B_gblurON_ksize5_savex_20250515_132346",
    # "gray_Phi-3.5_gblurON_ksize5_savex_20250515_130624",
    # "gray_crossattack_phi_llama_qwen_gblurON_ksize5_MA_savex_20250514_140130",
    # "gray_crossattack_phi_llama_qwen_0.4_20250516_135203", # localization
    # "gray_crossattack_phi_qwen_llava_20250516_190506", # localization
    # "gray_Llama_localization_eps0.4_smaller_crop_init_20250516_132017",
    # "gray_Llama_localization_eps0.4_20250516_130521",
    # "gray_Llama_localization_continue_eps0.5_20250515_174359",
    # "gray_Llama_localization_MA_20250514_140414",
    # "gray_Llama_localization_continue_20250514_142617",
    # "gray_Llama_localization_20250513_202253"
    # "gray_crossattack_llama_qwen_llava_localization_20250518_010005",
    # "gray_crossattack_phi_llama_llava_localization_20250518_012640",
    # "gray_crossattack_phi_llama_qwen_llava_20250518_203800",
    "gray_crossattack_phi_llama_qwen_llava_MA_20250518_202524"

]

runs_path = "/home/jovyan/rahmatullaev/adversarial/runs/"

judge_model_name = "google/gemma-3-4b-it"
device = "cuda:0"
# dtype = 'bfloat16'
dtype = 'float32'

checker = SafetyChecker(judge_model_name, device, dtype)
BATCH_SIZE = 16  # how many texts to process per model inference call

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def parse_iteration_number(filename):
    """
    Extract iteration as int from filenames like: test_results_iter_3000.csv
    Example: parse_iteration_number("test_results_iter_3000.csv") -> 3000
    """
    # Remove prefix + suffix
    # One approach is to do:
    # "test_results_iter_3000.csv".replace("test_results_iter_", "").replace(".csv", "")
    # Then cast to int.
    base = filename.replace("test_results_iter_", "").replace(".csv", "")
    return int(base)

for exp in exps:
    # Only load CSV files that start with "test_results_iter_" to avoid old metric files:
    try:
        iter_list = [
            x for x in os.listdir(os.path.join(runs_path, exp))
            if x.startswith("test_results_iter_") and x.endswith(".csv")
        ]

        # We'll store tuples of (iteration_number, model_name, unsafe_count) for final pivot
        data_rows = []
        
        # Create a directory for detailed safety evaluations
        safety_details_dir = os.path.join(runs_path, exp, "safety_details")
        os.makedirs(safety_details_dir, exist_ok=True)
        os.makedirs(os.path.join(safety_details_dir, judge_model_name.split("/")[-1]), exist_ok=True)

        for filename in tqdm(iter_list, desc=f"Processing {exp}"):
            csv_path = os.path.join(runs_path, exp, filename)
            iter_data = pd.read_csv(csv_path)

            # parse iteration from the filename
            iteration_num = parse_iteration_number(filename)

            # check if the safety details file already exists
            safety_details_path = os.path.join(safety_details_dir, judge_model_name.split("/")[-1], f"safety_details_iter_{iteration_num}.csv")
            if os.path.exists(safety_details_path):
                print(f"Skipping {filename} because it already exists")
                continue

            # We'll read the questions from the first column
            questions = iter_data.iloc[:, 0].tolist()

            # model_columns: skip first column (the question)
            model_columns = iter_data.columns[1:]  

            # We'll keep track of total "unsafe" occurrences for each model
            unsafe_counts = defaultdict(int)
            
            # Create a list to store detailed safety evaluation results
            safety_details = []

            # For each column (each model), gather all texts and process in batches
            for m_col in model_columns:
                # Extract all texts from that column
                texts = iter_data[m_col].fillna("").astype(str).tolist()
                total_texts = len(texts)
                column_unsafe = 0

                # We'll create prompts in batches
                for batch_index, batch in enumerate(chunk_list(texts, BATCH_SIZE)):
                    # Build prompts for this batch
                    prompts = []
                    original_texts = []
                    for j, text_str in enumerate(batch):
                        # Calculate row index among the entire set
                        row_idx = batch_index * BATCH_SIZE + j
                        question = questions[row_idx]

                        # Remove the question from text if needed
                        if question in text_str:
                            q_end = text_str.find(question) + len(question)
                        else:
                            q_end = 0

                        prompt = text_str[q_end:]
                        prompts.append(prompt)
                        original_texts.append(text_str)

                    inputs = prompts
                    with torch.no_grad(): 
                        result = checker.evaluate_text(inputs, max_tokens=10)

                    # Decode each answer and store detailed results
                    for idx, (output, ans_text) in enumerate(zip(result, prompts)):
                        row_idx = batch_index * BATCH_SIZE + idx
                        question_text = questions[row_idx] if row_idx < len(questions) else ""
                        
                        safety_details.append({
                            "question": question_text,
                            "model": m_col,
                            "text_answer": ans_text,
                            "is_safe": output.is_safe,
                            "safety_status": "safe" if output.is_safe else "unsafe"
                        })
                        
                        if not output.is_safe:
                            column_unsafe += 1

                # Now that we processed the entire column in batches, compute fraction
                unsafe_count_for_model = column_unsafe / total_texts

                # Store the result
                data_rows.append((iteration_num, m_col, unsafe_count_for_model))

            # Save detailed safety evaluation results for this iteration
            safety_df = pd.DataFrame(safety_details)
            safety_df.to_csv(safety_details_path, sep=";", index=False)

            # Optionally compute mean across all models in this CSV
            # i.e., average the unsafe_count among model_columns
            # We'll do that by summing the last N = len(model_columns) entries.
            sum_unsafe = 0
            for m_col in model_columns:
                sum_unsafe += data_rows[-len(model_columns):][model_columns.tolist().index(m_col)][2]

            mean_value = sum_unsafe / len(model_columns)
            data_rows.append((iteration_num, "ALL_MODELS_MEAN", mean_value))

        # Build a DataFrame from the collected rows
        df = pd.DataFrame(data_rows, columns=["iteration", "model", "unsafe_count"])

        # We want rows = iteration, columns = model, so pivot:
        pivot_df = df.pivot(index="iteration", columns="model", values="unsafe_count")

        # Sort by iteration
        pivot_df = pivot_df.sort_index()

        # Save to CSV with rows = iteration, columns = model
        output_csv_path = os.path.join(runs_path, exp, "unsafe_metrics_models.csv")
        pivot_df.to_csv(output_csv_path, index=True)

        # Find best iteration among "ALL_MODELS_MEAN"
        # We can query pivot_df["ALL_MODELS_MEAN"] directly if that column exists
        if "ALL_MODELS_MEAN" in pivot_df.columns:
            best_iter = pivot_df["ALL_MODELS_MEAN"].idxmax()
            print(f"EXPERIMENT {exp}, BEST ITER {best_iter} with ASR (mean): {pivot_df['ALL_MODELS_MEAN'].loc[best_iter]}")
            with open(os.path.join(runs_path, exp, "best.txt"), "w") as f:
                f.write(str(best_iter))
            with open(os.path.join(runs_path, exp, 'safety_details', judge_model_name.split("/")[-1], "best_iter.txt"), "w") as f:
                f.write(str(best_iter))
        else:
            print(f"EXPERIMENT {exp}, no ALL_MODELS_MEAN column found.")

        # Now we plot: each column is a line, iteration on the x-axis
        plt.figure(figsize=(12, 6))
        for col in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[col], marker='o', label=col)

        plt.title(f"Unsafe Count for Each Model ({exp})")
        plt.xlabel("Iteration (numeric)")
        plt.ylabel("Unsafe Count")
        xticks = list(range(pivot_df.index.min(), pivot_df.index.max() + 1, 500))
        plt.xticks(xticks, rotation=60)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(runs_path, exp, "unsafe_metrics_models_plot.png")
        plot_path = os.path.join(runs_path, exp, 'safety_details', judge_model_name.split("/")[-1], "unsafe_metrics_models_plot.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"ERROR {e} for {exp}")
        continue