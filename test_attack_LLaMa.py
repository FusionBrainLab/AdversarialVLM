from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from attack_list_o1upd_llamaV import setup_device
from src.llama32processor import load_model_and_processor# , DifferentiableMllamaImageProcessor,

from questions import not_safe_questions_test, questions

model, processor = load_model_and_processor("alpindale/Llama-3.2-11B-Vision-Instruct", torch.device("cuda:7"))

image_path = "/home/jovyan/rahmatullaev/adversarial/runs/airi_tanh_Llama3.2-3-11B-Vision_start_added_unsafe_questions_20241112_175855/optimized_image_iter_2001.png"

original_image = Image.open(image_path)
original_tensor = np.fromfile(image_path[:-4] + ".bin")

for question in questions[:10]+not_safe_questions_test:
    inference_prompts1 = [processor.apply_chat_template([
            {
                "role": "user", 
                "content": 
                    [
                        {"type": "image"}, 
                        {"type": "text", "text": question}
                    ]
            },
        ], add_generation_prompt=True)]

    inputs_for_inference = processor(
                    text=inference_prompts1, 
                    images=[original_image], 
                    return_tensors="pt", 
                    padding=True
                ).to(torch.device("cuda:7"))

    outputs_inference = model.generate(**inputs_for_inference, max_new_tokens=64, do_sample=False)
    generated_text = processor.tokenizer.decode(outputs_inference[0], skip_special_tokens=False)
    print(generated_text)
    print("_________________________\n\n\n")