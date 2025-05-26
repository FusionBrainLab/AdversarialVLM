from typing import List, Tuple, Union
import torch
import numpy as np
import random

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.image_processing_utils import BatchFeature
import torch.nn.functional as F
from torchvision.transforms import functional as F_tv


def load_model_and_processor(model_name, device):
    """Load the model and processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    return model, processor

class AdvGemma3Inputs:
    def __init__(self, questions, test_questions, batch_size, original_image, processor, device="cuda:0", target_text="sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.original_image = original_image
        self.device = device
        
        # self.extra_token = "<|im_end|>\n"
        # self.shift = len(processor.tokenizer.encode(self.extra_token))

        # if isinstance(target_text, list):
        #     self.target_texts = target_text  # Храним весь список
        #     self.target_text = target_text[0]  # Начальное значение
        # else:
        #     self.target_texts = [target_text]
        #     self.target_text = target_text
        
        # self.update_target_tokens()

    def update_target_tokens(self):
        raise NotImplementedError("pqlet: Gemma3 processor does not support training. Used only for evaluation.")
        self.target_tokens = self.processor.tokenizer(self.target_text+self.extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.suffix_length = self.target_tokens.shape[1]
        self.target = self.target_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)
    
    def set_target_text(self, target_text):
        raise NotImplementedError("pqlet: Gemma3 processor does not support training. Used only for evaluation.")
        self.target_text = target_text
        self.update_target_tokens()

    def get_loss(self, logits):
        raise NotImplementedError("pqlet: Gemma3 processor does not support training. Used only for evaluation.")
        # Extract relevant logits and compute loss
        logits_suffix = logits[:, -self.suffix_length:-self.shift, :]
        logits_suffix = logits_suffix.permute(0, 2, 1)
        loss = F.cross_entropy(logits_suffix, self.target)
        return loss

    def get_inputs_train(self):
        raise NotImplementedError("pqlet: Gemma3 processor does not support training. Used only for evaluation.")
        batch_questions = random.choices(self.questions, k=self.batch_size)
        prompts = [self.processor.apply_chat_template([
            {
                "role": "user", 
                "content": 
                    [
                        {"type": "image"}, 
                        {"type": "text", "text": q}
                    ]
            },
            {
                "role": "assistant",
                "content": 
                    [
                        {"type": "text", "text": self.target_text}
                    ]
            }
        ]) for q in batch_questions]
        inputs = self.processor(
            text=prompts,
            images=[self.original_image for _ in batch_questions],
            padding=True,
            return_tensors="pt",
        ).to(torch.device(self.device))
        return inputs
        
    def get_inputs_inference(self, img, question = None):
        if question is None:
            question = self.test_questions[0]
        inference_prompts = [self.processor.apply_chat_template([
                {
                    "role": "user", 
                    "content": 
                        [
                            {"type": "image"}, 
                            {"type": "text", "text": question}
                        ]
                },
            ], add_generation_prompt=True)]
            
        inputs_for_inference = self.processor(
                text=inference_prompts, 
                images=[img], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
        
        return inputs_for_inference


# class DifferentiableGemma3ImageProcessor():
#     raise NotImplementedError("pqlet: Gemma3 processor does not support training. Used only for evaluation.")