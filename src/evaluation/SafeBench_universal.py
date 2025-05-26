import os
import sys
import argparse

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

######################################

def main(exp, model_suf, device):
    assert model_suf in dict_suf_modelname, "Model suffix not found. Shall be one of: " + ", ".join(dict_suf_modelname.keys())
    model_name = dict_suf_modelname[model_suf]
    load_model_and_processor, AdvInputs, DIProcessor = load_components(model_name, )
    model, processor = load_model_and_processor(model_name, torch.device(device))
    paths = [f"/home/jovyan/rahmatullaev/adversarial/tests/{exp[0]}_{exp[1]}/{model_suf}"]
    images = [f"/home/jovyan/rahmatullaev/adversarial/runs/{exp[0]}/optimized_image_iter_{exp[1]+1}.png"]

    # Iterate over list of length 1 - from legacy code
    for path, image in zip(paths, images):
        if not os.path.exists(image):
            print("ERROR, image doesn't exist:", image)
        os.makedirs(path, exist_ok=True)
        
        save_path = path
        original_image = Image.open(image)
        
        for i in tqdm(range(1, 24)):
            sfb_i = pd.read_csv(f"{sfb_path}/{i}.csv", names=[0])
            generated_texts = []

            batch_size = 1
            batches = [sfb_i.iloc[j:j+batch_size] for j in range(0, len(sfb_i), batch_size)]

            for batch in batches:
                if model_suf == "phi35":
                    inference_prompts = [
                        f'<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>\n' for text in batch[0]
                    ]
                    inputs = batch_processing(
                        processor, 
                        inference_prompts, 
                        [original_image for _ in batch[0]]
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
                                        {"type": "text", "text": question}
                                    ]
                            },
                        ], add_generation_prompt=True) for question in batch[0]]

                    inputs = processor(
                            text=inference_prompts, 
                            images=[original_image for _ in batch[0]], 
                            return_tensors="pt", 
                            padding=True
                        ).to(device)

                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, min_new_tokens=32)
                for idx, output in enumerate(outputs):
                    ans = processor.decode(output[inputs["input_ids"].shape[1]:],  skip_special_tokens=True)
                    generated_texts.append(ans)
            sfb_answers = pd.DataFrame(generated_texts)
            sfb_answers.to_csv(os.path.join(save_path, f"{i}.csv"),index=False, header=False)

######################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adversarial Model Testing")
    parser.add_argument("--exp", type=str, help="Experiment name")
    parser.add_argument("--iter", type=int, help="Iteration number")
    parser.add_argument("--model_suf", type=str, help="What model to use for inferencing the adversarial image")
    parser.add_argument("--cuda_num", type=int, help="Cuda number")

    args = parser.parse_args()
    exp = [args.exp, args.iter]
    main(exp, args.model_suf, args.cuda_num)