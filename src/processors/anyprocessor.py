from typing import List, Tuple, Union, Dict, Any
import PIL
import torch
import numpy as np
import random
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoModelForImageTextToText,
    AutoTokenizer, 
    BaseImageProcessorFast
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
import torch.nn.functional as F
from PIL import Image

def load_model_and_processor(model_name: str, device: str = "cuda:0"):
    """Load the model and processor for any supported model."""
    
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left', trust_remote_code=True, use_fast=True)
    if not getattr(processor, "apply_chat_template", False):
        raise ValueError("processor does not have apply_chat_template")
        
    if not getattr(processor, "image_processor", False):
        raise ValueError("processor does not have image_processor")
        
    if not isinstance(processor.image_processor, BaseImageProcessorFast):
        raise ValueError("processor.image_processor is not BaseImageProcessorFast. May be you need to unpdate transformers library.")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
    except Exception:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(device)
        except Exception as e:
            raise ValueError(f"Model {model_name} not supported: {model_name} is not AutoModelForCausalLM and is not AutoModelForImageTextToText. May be you need to update transformers library. Error: {e}")
    
    return model, processor

def auto_detect_extra_tokens(processor: AutoProcessor) -> str:
    """
    Автоматически определяет extra_token для модели на основе её названия и анализа токенизатора
    """
    try:        
        inputs = processor.tokenizer.apply_chat_template(
            {"role": "user", "content": [{"type": "text", "text": "Say Yes."}]}, 
            add_generation_prompt=True, tokenize=False
        )
        
        extra_tokens = inputs[len("Yes.") + inputs.find("Yes."):]
        
        return extra_tokens
        
    except Exception as e:
        print(f"Ошибка при определении extra_token: {e}")
        return ''

class AdvAnyInputs:
    """
    Универсальный класс для обработки входных данных для любых моделей
    """
    
    refuses = [
        "As an AI language model, I am not",
        "As an AI language model, I do not",
        "As an AI, I am not",
        "As an AI, I do not",
        "As an AI",
        "I'm sorry, but I am not",
        "I'm sorry, but I cannot",
        "I cannot provide",
        "It is not",
        "I cannot provide",
        "I can't help",
        "I can't support",
        "I can't provide",
        "I can't prescribe",
        "I can't advise",
        "I can't accommodate",
        "I can't diagnose",
        "I can't engage",
        "I can't fulfill",
        "I can't recommend",
        "I can't assist",
        "I can't answer",
        "Absolutely not.",
        "As an AI language model, I am not",
        "As an AI language model, I do not",
        "As an AI language model, I cannot",
        "I'm sorry, but"
        "I'm sorry, but I can't",
        "I'm sorry, but I cannot",
        "No, it is not",
        "Sorry, I can't",
        "Sorry, I cannot",
        "Sorry, but I can't assist with that",
        "As an AI, I don't",
        "As an AI, I'm not",
        "Sorry, but I cannot help you"
    ]
    
    def __init__(self, questions: List[str], test_questions: List[str], batch_size: int, 
                 original_image: PIL.Image.Image, processor: AutoProcessor, device: str = "cuda:0", target_text: Union[str, List[str]] = "sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.original_image = original_image
        self.device = device
        
        # Автоматически определяем extra_token
        self.extra_token = auto_detect_extra_tokens(processor)
        self.shift = len(processor.tokenizer.encode(self.extra_token, add_special_tokens=False)) if self.extra_token else 2 

        if isinstance(target_text, list):
            self.target_texts = target_text
            self.target_text = target_text[0]
        else:
            self.target_texts = [target_text]
            self.target_text = target_text
        
        # Initialize refuse text for DPO
        self.refuse_text = random.choice(self.refuses)
        
        self.update_target_tokens()
        self.update_refuse_tokens()

    def update_target_tokens(self):
        full_target = self.target_text + self.extra_token if self.extra_token else self.target_text
        
        self.target_tokens = self.processor.tokenizer(
            full_target, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        self.suffix_length = self.target_tokens.shape[1]
        
        # Удаляем extra_token из target для вычисления loss
        if self.shift > 0:
            self.target = self.target_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)
        else:
            self.target = self.target_tokens.repeat(self.batch_size, 1).to(self.device)
    
    def set_target_text(self, target_text: str):
        self.target_text = target_text
        self.update_target_tokens()

    def get_loss(self, logits):
        # Extract relevant logits and compute loss
        end_pos = -self.shift if self.shift > 0 else logits.shape[1]
        logits_suffix = logits[:, -self.suffix_length:end_pos, :]
        logits_suffix = logits_suffix.permute(0, 2, 1)
        
        loss = F.cross_entropy(logits_suffix, self.target)
        return loss

    def get_inputs_train(self):
        batch_questions = random.choices(self.questions, k=self.batch_size)
        
        # Пробуем использовать chat_template
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
            images=[[self.original_image] for _ in batch_questions],
            padding=True,
            return_tensors="pt",
        ).to(torch.device(self.device))
        
        return inputs
        
    def get_inputs_inference(self, img: PIL.Image.Image, question: str = None):
        if question is None:
            question = self.test_questions[0]
            
        # Пробуем использовать chat_template
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
    
    def update_refuse_tokens(self):
        """Update refuse tokens when refuse_text changes."""
        full_refuse = self.refuse_text + self.extra_token if self.extra_token else self.refuse_text
        
        self.refuse_tokens = self.processor.tokenizer(
            full_refuse, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        self.refuse_suffix_length = self.refuse_tokens.shape[1]
        
        # Удаляем extra_token из refuse_target для вычисления loss
        if self.shift > 0:
            self.refuse_target = self.refuse_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)
        else:
            self.refuse_target = self.refuse_tokens.repeat(self.batch_size, 1).to(self.device)

    def set_refuse_text(self, refuse_text):
        """Set refuse text for DPO negative samples."""
        self.refuse_text = refuse_text
        self.update_refuse_tokens()
    
    def get_inputs_refuse(self):
        """Get inputs with refuse text as target."""
        batch_questions = random.choices(self.questions, k=self.batch_size)
        
        # Пробуем использовать chat_template
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
                        {"type": "text", "text": self.refuse_text}
                    ]
            }
        ]) for q in batch_questions]
        
        inputs = self.processor(
            text=prompts,
            images=[[self.original_image] for _ in batch_questions],
            padding=True,
            return_tensors="pt",
        ).to(torch.device(self.device))
        
        return inputs
    
    def get_loss_refuse(self, logits):
        """Compute loss for refuse text."""
        end_pos = -self.shift if self.shift > 0 else logits.shape[1]
        logits_suffix = logits[:, -self.refuse_suffix_length:end_pos, :]
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
        repeat_size = len(adv_pixel_values.shape)*[1]
        repeat_size[0] = self.batch_size
        adv_pixel_values_repeated = adv_pixel_values.repeat(repeat_size)
        ref_pixel_values_repeated = ref_pixel_values.repeat(repeat_size)
        
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
        end_pos_target = -self.shift if self.shift > 0 else logits_pos_pi.shape[1]
        end_pos_refuse = -self.shift if self.shift > 0 else logits_neg_pi.shape[1]
        
        logits_pos_pi_suffix = logits_pos_pi[:, -self.suffix_length:end_pos_target, :]
        logits_neg_pi_suffix = logits_neg_pi[:, -self.refuse_suffix_length:end_pos_refuse, :]
        logits_pos_ref_suffix = logits_pos_ref[:, -self.suffix_length:end_pos_target, :]
        logits_neg_ref_suffix = logits_neg_ref[:, -self.refuse_suffix_length:end_pos_refuse, :]
        
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
        # Вычисление log-mixture для отрицательных
        #   log π_mix(y_l) = log(λ·exp(log_pi_neg) + (1-λ)·exp(log_ref_neg))
        log_mix_neg = torch.log(
            lambda_ * torch.exp(log_pi_neg) +
            (1 - lambda_) * torch.exp(log_ref_neg)
        )

        # BDPO-advantage и loss
        #    advantage = β·(log_pi_pos - log_mix_neg) - β·(log_ref_pos - log_ref_neg)
        advantage = beta * (log_pi_pos - log_mix_neg) - beta * (log_ref_pos - log_ref_neg)
        bdpo_loss = -F.logsigmoid(advantage)

        return bdpo_loss.mean()

class DifferentiableAnyImageProcessor:
    def __init__(self, orig_processor: AutoProcessor, device: str = "cuda:0"):
        self.orig_processor = orig_processor
        if isinstance(self.orig_processor, BaseImageProcessorFast):
            self.orig_processor = self.orig_processor
        else:
            raise ValueError("processor.image_processor is not BaseImageProcessorFast. May be you need to unpdate transformers library.")
        self.device = device
        self.do_convert_rgb = getattr(orig_processor, "do_convert_rgb", False)
        self.crop_size = getattr(orig_processor, "crop_size", None)
        
    def process(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process the input image tensor using fast processor with gradient preservation
        """
        processed = self.orig_processor(
            [image], 
            return_tensors="pt"
        )
        
        # Сохраняем градиенты
        if image.requires_grad:
            processed["pixel_values"].retain_grad()
        
        return processed
        
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
            tensor_image, _ = self.orig_processor.resize(tensor_image)
        return tensor_image.to(self.device)

    def tensor2pil(self, tensor: torch.Tensor) -> PIL.Image.Image:
        """
        Convert a tensor image to a PIL image.
        """
        return self.orig_processor.post_process(tensor)
    
    def tensor2pil(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x.clamp(0, 1)
        img = (x * 255).cpu().detach().permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img