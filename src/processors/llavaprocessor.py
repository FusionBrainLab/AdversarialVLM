import random
from typing import Tuple, Union
import PIL
import torch
import numpy as np
from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
import torch.nn.functional as F
from PIL import Image


def load_model_and_processor(model_name: str, device: str):
    """Load the model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16
    ).to(device)
    
    processor = LlavaProcessor.from_pretrained(model_name)
    return model, processor


class AdvLlavaInputs:
    refuses = [
        "As an AI language model, I am not",
        "As an AI language model, I do not",
        "As an AI, I am not",
        "As an AI, I do not",
        "As an AI",
        "I'm sorry, but I am not",
        "I'm sorry, but I cannot",
        "I's sorry", # it is not bug, llava really answers this way
        "It is not"
    ]

    def __init__(
            self, 
            questions: list, 
            test_questions: list, 
            batch_size: int, 
            original_image: PIL.Image.Image, 
            processor, 
            device: str = "cuda:0", 
            target_text: Union[str, list] = "sure, here it is!"
        ):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.original_image = original_image
        self.device = device
        
        self.extra_token = "</s>"
        self.shift = len(processor.tokenizer.encode(self.extra_token)) # first token is extra
        
        if isinstance(target_text, list):
            self.target_texts = target_text  # Храним весь список
            self.target_text = target_text[0]  # Начальное значение
        else:
            self.target_texts = [target_text]
            self.target_text = target_text
        
        # Initialize refuse text for DPO
        self.refuse_text = random.choice(self.refuses)
        
        self.update_target_tokens()
        self.update_refuse_tokens()
        
    def update_target_tokens(self):
        self.target_tokens = self.processor.tokenizer(self.target_text+self.extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.suffix_length = self.target_tokens.shape[1]
        self.target = self.target_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)
    
    def set_target_text(self, target_text):
        self.target_text = target_text
        self.update_target_tokens()
    
    def update_refuse_tokens(self):
        """Update refuse tokens when refuse_text changes."""
        self.refuse_tokens = self.processor.tokenizer(self.refuse_text+self.extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.refuse_suffix_length = self.refuse_tokens.shape[1]
        self.refuse_target = self.refuse_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)

    def set_refuse_text(self, refuse_text):
        """Set refuse text for DPO negative samples."""
        self.refuse_text = refuse_text
        self.update_refuse_tokens()
    
    def get_loss(self, logits):
        # Extract relevant logits and compute loss
        logits_suffix = logits[:, -self.suffix_length:-self.shift, :]
        logits_suffix = logits_suffix.permute(0, 2, 1)
        loss = F.cross_entropy(logits_suffix, self.target)
        return loss

    def get_inputs_train(self):
        batch_questions = random.choices(self.questions, k=self.batch_size)
        
            
        prompts = [self.processor.apply_chat_template([
            {
                "role": "user",
                "content": 
                    [
                        {"type": "text", "text": q},
                        {"type": "image"},
                    ],
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
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
            }
        ], add_generation_prompt=True)]
        
        inputs_for_inference = self.processor(
                text=inference_prompts, 
                images=[img], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

        return inputs_for_inference

    def get_inputs_refuse(self):
        """Get inputs with refuse text as target."""
        batch_questions = random.choices(self.questions, k=self.batch_size)
        self.set_refuse_text(random.choice(self.refuses))
            
        prompts = [self.processor.apply_chat_template([
            {
                "role": "user",
                "content": 
                    [
                        {"type": "text", "text": q},
                        {"type": "image"},
                    ],
            },
            {
                "role": "assistant",
                "content": 
                    [
                        {"type": "text", "text": self.refuse_text}
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
    
    def get_loss_refuse(self, logits):
        """Compute loss for refuse text."""
        logits_suffix = logits[:, -self.refuse_suffix_length:-self.shift, :]
        logits_suffix = logits_suffix.permute(0, 2, 1)
        loss = F.cross_entropy(logits_suffix, self.refuse_target)
        return loss
    
    def compute_dpo_loss(self, model, adv_pixel_values, ref_pixel_values, beta=0.3, lambda_=1.0):
        """
        Compute BDPO loss using same model with adversarial vs original image.
        
        Args:
            model: The model π_θ (same for policy and reference)
            adv_pixel_values: Adversarial image tensor (x_0 + x)
            ref_pixel_values: Original image tensor (x_0) 
            beta: DPO temperature parameter (typically 0.1-0.5)
            lambda_: BDPO mixture parameter (default 1.0 = standard DPO)
            
        Returns:
            BDPO loss tensor
        """
        # Get inputs for positive (target) and negative (refuse) examples
        inputs_pos = self.get_inputs_train()
        inputs_neg = self.get_inputs_refuse()
        
        # Add adversarial pixel values for policy
        # repeat_size = len(adv_pixel_values.shape)*[1]
        # repeat_size[0] = self.batch_size
        adv_pixel_values_repeated = adv_pixel_values# .repeat(repeat_size)
        ref_pixel_values_repeated = ref_pixel_values# .repeat(repeat_size)
        
        inputs_pos['pixel_values'] = adv_pixel_values_repeated
        inputs_neg['pixel_values'] = adv_pixel_values_repeated
        
        # ---------------------- Policy forward pass (adversarial image) ----------------------
        logits_pos_pi = model(**inputs_pos).logits[:, :-1, :]
        logits_neg_pi = model(**inputs_neg).logits[:, :-1, :]
        
        # ---------------------- Reference forward pass (original image, no grad) ----------------------
        with torch.no_grad():
            inputs_pos_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_pos.items()}
            inputs_neg_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_neg.items()}
            inputs_pos_ref['pixel_values'] = ref_pixel_values_repeated
            inputs_neg_ref['pixel_values'] = ref_pixel_values_repeated
            
            logits_pos_ref = model(**inputs_pos_ref).logits[:, :-1, :]
            logits_neg_ref = model(**inputs_neg_ref).logits[:, :-1, :]
        
        # ---------------------- Extract relevant logits suffixes ----------------------
        logits_pos_pi_suffix = logits_pos_pi[:, -self.suffix_length:-self.shift, :]
        logits_neg_pi_suffix = logits_neg_pi[:, -self.refuse_suffix_length:-self.shift, :]
        logits_pos_ref_suffix = logits_pos_ref[:, -self.suffix_length:-self.shift, :]
        logits_neg_ref_suffix = logits_neg_ref[:, -self.refuse_suffix_length:-self.shift, :]
        
        # ---------------------- Log probabilities ----------------------
        log_probs_pos_pi = F.log_softmax(logits_pos_pi_suffix, dim=-1)
        log_probs_neg_pi = F.log_softmax(logits_neg_pi_suffix, dim=-1)
        log_probs_pos_ref = F.log_softmax(logits_pos_ref_suffix, dim=-1)
        log_probs_neg_ref = F.log_softmax(logits_neg_ref_suffix, dim=-1)
        
        # ---------------------- Gather target token probabilities ----------------------
        # Ensure we only sum over valid tokens
        min_len_pos = min(log_probs_pos_pi.shape[1], self.target.shape[1])
        min_len_neg = min(log_probs_neg_pi.shape[1], self.refuse_target.shape[1])
        
        # log π_θ(y|x_adv) - policy with adversarial image
        log_pi_pos = torch.gather(
            log_probs_pos_pi[:, :min_len_pos, :], 
            2, 
            self.target[:, :min_len_pos].unsqueeze(-1)
        ).squeeze(-1).sum(dim=1)
        
        log_pi_neg = torch.gather(
            log_probs_neg_pi[:, :min_len_neg, :], 
            2, 
            self.refuse_target[:, :min_len_neg].unsqueeze(-1)
        ).squeeze(-1).sum(dim=1)
        
        # log π_θ(y|x_orig) - reference with original image
        log_ref_pos = torch.gather(
            log_probs_pos_ref[:, :min_len_pos, :], 
            2, 
            self.target[:, :min_len_pos].unsqueeze(-1)
        ).squeeze(-1).sum(dim=1)
        
        log_ref_neg = torch.gather(
            log_probs_neg_ref[:, :min_len_neg, :], 
            2, 
            self.refuse_target[:, :min_len_neg].unsqueeze(-1)
        ).squeeze(-1).sum(dim=1)
        
        # ---------------------- BDPO objective ----------------------
        # Вычисление log-mixture для отрицательных с численной стабильностью
        # Используем log-sum-exp трюк для предотвращения underflow/overflow
        #   log π_mix(y_l) = log(λ·exp(log_pi_neg) + (1-λ)·exp(log_ref_neg))
        
        # Стабильная версия log-sum-exp
        max_log_neg = torch.max(log_pi_neg, log_ref_neg)
        log_mix_neg = max_log_neg + torch.log(
            lambda_ * torch.exp(log_pi_neg - max_log_neg) +
            (1 - lambda_) * torch.exp(log_ref_neg - max_log_neg)
        )

        # BDPO-advantage и loss
        #    advantage = β·(log_pi_pos - log_mix_neg) - β·(log_ref_pos - log_ref_neg)
        advantage = beta * (log_pi_pos - log_mix_neg) - beta * (log_ref_pos - log_ref_neg)
        
        # Проверка на NaN и inf для отладки
        if torch.isnan(advantage).any() or torch.isinf(advantage).any():
            print(f"Warning: advantage contains NaN or inf values!")
            print(f"log_pi_pos range: [{log_pi_pos.min():.4f}, {log_pi_pos.max():.4f}]")
            print(f"log_mix_neg range: [{log_mix_neg.min():.4f}, {log_mix_neg.max():.4f}]")
            print(f"log_ref_pos range: [{log_ref_pos.min():.4f}, {log_ref_pos.max():.4f}]")
            print(f"log_ref_neg range: [{log_ref_neg.min():.4f}, {log_ref_neg.max():.4f}]")
            # Заменяем NaN и inf на большие отрицательные значения для стабильности
            advantage = torch.where(torch.isnan(advantage) | torch.isinf(advantage), 
                                  torch.full_like(advantage, -100.0), advantage)
        
        bdpo_loss = -F.logsigmoid(advantage)

        return bdpo_loss.mean()

class DifferentiableLlavaImageProcessor():
    def __init__(self, orig_processor, device):        
        self.image_mean = torch.tensor(orig_processor.image_mean).view(-1, 1, 1).to(device)
        self.image_std = torch.tensor(orig_processor.image_std).view(-1, 1, 1).to(device)
        self.do_convert_rgb = orig_processor.do_convert_rgb
        self.crop_size = orig_processor.crop_size
        self.device = device
    
    def process(self, image: torch.Tensor) -> dict:
        new_h, new_w = self.crop_size["height"], self.crop_size["width"]
        image = F.interpolate((image).unsqueeze(0), size=[new_h, new_w], mode='bilinear', align_corners=False, antialias=True)
        image = image.squeeze(0)
        image_transformed = (image - self.image_mean) / self.image_std
        data = {
            "pixel_values": image_transformed.unsqueeze(0)
        }
        return data
    
    def tensor2pil(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x.clamp(0, 1)
        img = (x * 255).cpu().detach().permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img
    
    def pil_to_tensor(self, image: PIL.Image, resize: bool = False) -> torch.Tensor:
        """
        Convert a PIL image to a tensor.

        Args:
            image: PIL image
            resize: Whether to resize the image to the optimal size for the model.

        Returns:
            A tensor image, shape (3, H, W)
        """
        if self.do_convert_rgb:
            image = image.convert("RGB")
        tensor_image = torch.tensor(np.array(image).astype(np.float32) / 255).permute(2, 0, 1)

        if resize:
            # num_channels, height, width = image.shape
            # tensor_image = tensor_image.reshape((1, num_channels, height, width))
            tensor_image, _ = self.image_processor.resize(tensor_image)
        return tensor_image.to(self.device)

    # def pil_to_tensor(self, image: PIL.Image, resize: bool = False) -> torch.Tensor:
    #     image = image.convert("RGB")
    #     if resize:
    #         image = self.fit_size_pil(image)
    #     return torch.tensor(np.array(image).astype(np.float32) / 255).permute(2, 0, 1)
