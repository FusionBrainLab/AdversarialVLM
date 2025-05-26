import os
import sys
import argparse
import glob
import json
from pathlib import Path

project_root = os.path.abspath("/home/jovyan/rahmatullaev/adversarial/src")
sys.path.append(project_root)

from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.processors import load_components
from src.processors.phi3processor import batch_processing

sfb_path = "/home/jovyan/rahmatullaev/adversarial/SafeBench_Text/"
dict_suf_modelname = {
    "qwenVL": "Qwen/Qwen2-VL-2B-Instruct",
    "phi35": "microsoft/Phi-3.5-vision-instruct",
    "Llama32": "alpindale/Llama-3.2-11B-Vision-Instruct",
    "llava-hf": "llava-hf/llava-1.5-7b-hf",
}

def main(model_suf, device, image_type="SD_TYPO", result_dir=None):
    assert model_suf in dict_suf_modelname, "Model suffix not found. Shall be one of: " + ", ".join(dict_suf_modelname.keys())
    assert image_type in ["SD", "TYPO", "SD_TYPO"], "Image type must be one of: SD, TYPO, SD_TYPO"
    
    model_name = dict_suf_modelname[model_suf]
    load_model_and_processor, AdvInputs, DIProcessor = load_components(model_name)
    model, processor = load_model_and_processor(model_name, torch.device(device))
    
    if result_dir is None:
        result_dir = f"/home/jovyan/rahmatullaev/adversarial/tests/MM-SafetyBench_{image_type}/{model_suf}"
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # Get all categories (numbered directories)
    category_dirs = sorted([d for d in glob.glob("/home/jovyan/rahmatullaev/adversarial/datasets/MM_SafetyBench/images/*") 
                           if os.path.isdir(d) and os.path.basename(d).isdigit()],
                           key=lambda x: int(os.path.basename(x)))
    
    for category_dir in category_dirs:
        category_id = os.path.basename(category_dir)
        
        # Load the text prompts from JSON file
        json_file = f"/home/jovyan/rahmatullaev/adversarial/datasets/MM_SafetyBench/text/{category_id}.json"
        
        if not os.path.exists(json_file):
            print(f"Warning: Text file for category {category_id} not found. Skipping.")
            continue
        
        with open(json_file, 'r') as f:
            question_data = json.load(f)
        
        # Path to the specific image type directory for this category
        image_type_dir = os.path.join(category_dir, image_type)
        
        if not os.path.exists(image_type_dir):
            print(f"Warning: Image directory {image_type_dir} not found. Skipping.")
            continue
        
        # Get all image files in the directory
        image_files = sorted([f for f in glob.glob(f"{image_type_dir}/*.jpg") + glob.glob(f"{image_type_dir}/*.png")], 
                            key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # Collect all responses for this category
        all_responses = []
        
        for image_file in tqdm(image_files, desc=f"Processing category {category_id}"):
            image_id = os.path.basename(image_file).split('.')[0]
            
            # Skip if image_id is not in the question data
            if image_id not in question_data:
                print(f"Warning: No question data for image {image_id} in category {category_id}. Skipping.")
                continue
            
            # Get the rephrased question based on image type
            if image_type == "SD":
                prompt_key = "Rephrased Question(SD)"
            else:
                prompt_key = "Rephrased Question"
                
            if prompt_key not in question_data[image_id]:
                print(f"Warning: {prompt_key} not found for image {image_id} in category {category_id}. Using default key.")
                prompt_key = "Rephrased Question"
                
            prompt = question_data[image_id].get(prompt_key, "Describe this image.")
            
            try:
                original_image = Image.open(image_file)
                
                # Process the image and prompt based on the model
                if model_suf == "phi35":
                    inference_prompts = [
                        f'<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n'
                    ]
                    inputs = batch_processing(
                        processor, 
                        inference_prompts, 
                        [original_image]
                    ).to(device)
                    
                else:
                    if model_suf == "qwenVL":
                        if processor.chat_template is None:
                            processor.chat_template = processor.tokenizer.chat_template
                            
                    inference_prompts = [processor.apply_chat_template([
                            {
                                "role": "user", 
                                "content": 
                                    [
                                        {"type": "image"}, 
                                        {"type": "text", "text": prompt}
                                    ]
                            },
                        ], add_generation_prompt=True)]

                    inputs = processor(
                            text=inference_prompts, 
                            images=[original_image], 
                            return_tensors="pt", 
                            padding=True
                        ).to(device)

                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, min_new_tokens=32)
                
                # Process the model's response
                generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # Add the generated text to our collection
                all_responses.append(generated_text)
                
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                all_responses.append(f"ERROR: {str(e)}")
        
        # Save all responses for this category to a single CSV file with just the category number as filename
        csv_filename = os.path.join(result_dir, f"{category_id}.csv")
        pd.DataFrame(all_responses).to_csv(csv_filename, index=False, header=False)
        
        print(f"Saved {len(all_responses)} responses to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MM_SafetyBench Baseline Testing")
    parser.add_argument("--model_suf", type=str, required=True, help="What model to use for inferencing (phi35, qwenVL, Llama32, llava-hf)")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device number")
    parser.add_argument("--image_type", type=str, default="SD_TYPO", choices=["SD", "TYPO", "SD_TYPO"], 
                        help="Type of images to use (SD, TYPO, or SD_TYPO)")
    parser.add_argument("--result_dir", type=str, help="Directory to save results (default: tests/MM-Bench_{image_type}/{model_suf})")

    args = parser.parse_args()
    main(args.model_suf, args.cuda_num, args.image_type, args.result_dir) 