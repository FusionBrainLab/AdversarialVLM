from typing import List, Tuple, Union
import PIL
import torch
import numpy as np
import torchvision
import random

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.image_processing_utils import BatchFeature
import torch.nn.functional as F
from torchvision.transforms import functional as F_tv
from PIL import Image
import math


def load_model_and_processor(model_name, device):
    """Load the model and processor."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch.bfloat16
        )
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    return model, processor

class AdvQwen2VLInputs:
    refuses = [
        "As an AI language model, I cannot",
        "I'm sorry, but"
    ]
    def __init__(self, questions, test_questions, batch_size, original_image, processor, device="cuda:0", target_text="sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.original_image = original_image
        self.device = device
        
        self.extra_token = "<|im_end|>\n"
        self.shift = len(processor.tokenizer.encode(self.extra_token))

        if isinstance(target_text, list):
            self.target_texts = target_text  # Храним весь список
            self.target_text = target_text[0]  # Начальное значение
        else:
            self.target_texts = [target_text]
            self.target_text = target_text
        
        self.update_target_tokens()

    def update_target_tokens(self):
        self.target_tokens = self.processor.tokenizer(self.target_text+self.extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.suffix_length = self.target_tokens.shape[1]
        self.target = self.target_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)
    
    def set_target_text(self, target_text):
        self.target_text = target_text
        self.update_target_tokens()

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

class DifferentiableQwen2VLImageProcessor():
    def __init__(self, orig_processor, device):
        self.orig_processor = orig_processor
        self.device = device
        
        self.image_mean = torch.tensor(orig_processor.image_mean).view(-1, 1, 1).to(device)
        self.image_std = torch.tensor(orig_processor.image_std).view(-1, 1, 1).to(device)

        self.do_convert_rgb = orig_processor.do_convert_rgb
        self.patch_size = orig_processor.patch_size
        self.merge_size = orig_processor.merge_size
        self.min_pixels = orig_processor.min_pixels
        self.max_pixels = orig_processor.max_pixels
        self.temporal_patch_size = orig_processor.temporal_patch_size

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
            tensor_image = self.resize_tensor(tensor_image)
        return tensor_image.to(self.device)

    def resize_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize a tensor image so that it has the optimal size for the model.

        Args:
            image: tensor image, shape (3, H, W)

        Returns:
            Resized tensor image, shape (3, new_H, new_W)
        """
        new_h, new_w = self._optimal_size(image)
        image = F.interpolate(image.unsqueeze(0), size=[new_h, new_w], mode='bilinear', align_corners=False, antialias=True)
        image = image.squeeze(0)
        return image

    def tensor2pil(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x.clamp(0, 1)
        img = (x * 255).cpu().detach().permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def _optimal_size(self, image: Union[torch.Tensor]) -> Tuple[int, int]:
        """
        Calculates the optimal size for the image.

        Args:
            image: tensor image
        Returns:
            Tuple of new height and width
        """
        height, width = image.shape[1], image.shape[2]
        factor = self.patch_size * self.merge_size
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def pad(self, img: torch.Tensor):
        """
        Pads the input image tensor.

        Args:
            img: tensor image, shape (3, H, W)

        Returns:
            Padded tensor image, shape (3, H, W)
        """
        pass

    def process(self, image: torch.Tensor) -> dict:
        """
        Process the input image into a format compatible with the Qwen-VL model.

        Args:
            image (torch.Tensor): Input image tensor, shape (C, H, W)

        Returns:
            dict: A dictionary with the following keys:

                - pixel_values (torch.Tensor): Processed image tensor, shape (num_patches, channels * temporal_patch_size * patch_height * patch_width)
                - num_tiles (List[int]): Real number of tiles for the image, not padded.

        Usage example:
        
        adv_processor = DifferentiableQwen2VLImageProcessor(processor.image_processor)
        original_image = Image.open(os.path.join("./images", img_orig)).convert("RGB")
        x_0 = adv_processor.pil_to_tensor(original_image).to(device)
        inputs = processor(text=prompts, images=[original_image for _ in batch_questions], return_tensors="pt", padding=True).to(device)
        pixel_values = adv_processor.process(x_0)["pixel_values"]
        inputs['pixel_values'] = pixel_values
        """
        image = self.resize_tensor(image)
        image = (image - self.image_mean) / self.image_std
        c, h, w = image.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        # Add batch dimension
        image = image.unsqueeze(0)

        # Duplicate along the temporal dimension if the batch size is 1
        if image.shape[0] == 1:
            image = image.repeat(self.temporal_patch_size, 1, 1, 1)

        # Determine grid size
        grid_t = image.shape[0] // self.temporal_patch_size

        # Reshape to match the original implementation
        patches = image.reshape(
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Transpose to prepare for flattening
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

        # Flatten the patches
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, c * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return {
            "pixel_values": flatten_patches,
            "num_tiles": [grid_h * grid_w]
        }

    def process_batch(self, image_list: List[torch.Tensor]) -> dict:
        """
        Processes a batch of images into a format compatible with the Qwen-VL model.

        Args:
            image_list (List[torch.Tensor]): List of input image tensors, each with shape (C, H, W).

        Returns:
            dict: A dictionary with the following keys:
                - pixel_values (torch.Tensor): Batched processed image tensors, shape (batch_size, temporal_patch_size, num_patches, channels, patch_height, patch_width).
                - aspect_ratio_ids: None (aspect ratio ids not calculated at this stage).
                - num_tiles: Real number of tiles for each image in the batch.
        """
        pixel_values = []
        num_tiles = []
        for image in image_list:
            processed = self.process(image)
            pixel_values.append(processed["pixel_values"])
            num_tiles.append(processed["num_tiles"])
        pixel_values = torch.cat(pixel_values, dim=0)
        return {
            "pixel_values": pixel_values,
            "aspect_ratio_ids": None,
            "num_tiles": num_tiles
        }
