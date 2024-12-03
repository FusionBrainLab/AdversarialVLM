import random
from typing import List, Tuple, Union
import PIL
import torch
import numpy as np
import torchvision
from transformers.image_processing_utils import BatchFeature
import torch.nn.functional as F
from torchvision.transforms import functional as F_tv
from PIL import Image


class AdvMllamaInputs:
    def __init__(self, questions, test_questions, batch_size, original_image, processor, device="cuda:0", target_text="sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.target_text = target_text
        self.original_image = original_image
        self.device = device
        
        extra_token = None ## Your extra token, ex "<|eot_id|>"
        self.target_tokens = processor.tokenizer(target_text+extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        self.shift = len(processor.tokenizer.encode(extra_token)) # first token is extra
        self.suffix_length = self.target_tokens.shape[1]
        
        self.target = self.target_tokens[:, :-self.shift].repeat(batch_size, 1).to(self.device)
    
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
        if question is not None:
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


class DifferentiableAbstractImageProcessor():
    def __init__(self, orig_processor, device):
        """
        orig_processor : 
        
        """
        self.image_mean = torch.tensor(orig_processor.image_mean).view(-1, 1, 1).to(device)
        self.image_std = torch.tensor(orig_processor.image_std).view(-1, 1, 1).to(device)
        self.orig_processor = orig_processor
        self.device = device
        self.do_convert_rgb = orig_processor.do_convert_rgb

    
    def pil_to_tensor(self, image: PIL.Image, resize: bool = True) -> torch.Tensor:
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
            tensor_image, _ = self.resize_tensor(tensor_image)
        return tensor_image.to(self.device)

    def resize_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize a tensor image so that it has the optimal size for the model.

        Args:
            image: tensor image, shape (3, H, W)

        Returns:
            Resized tensor image, shape (3, new_H, new_W)
        """
        # C x H x W
        new_h, new_w = self._optimal_size(image)
        image = F.interpolate(image.unsqueeze(0), size=[new_h, new_w], mode='bilinear', align_corners=False)
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
        pass

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
        Process the input image into a format compatible with the LLaMA model.

        Args:
            image (torch.Tensor): Input image tensor, shape (C, H, W)

        Returns:
            dict: A dictionary with the following keys:

                - pixel_values (torch.Tensor): Processed image tensor, shape (1, 1, image_tiles, channels, tile_height, tile_width)
                - aspect_ratio_ids (np.ndarray): Aspect ratio ids for each image, shape (batch_size, max_num_images), padded to max_num_images with 0
                - num_tiles (List[List[int]]): Real number of tiles for each image, shape (batch_size, num_images_in_batch), not padded

        Usage example:
        
        adv_processor = DifferentiableAbstractImageProcessor(processor.image_processor)
        original_image = Image.open(os.path.join("./images", img_orig)).convert("RGB")
        x_0 = adv_processor.pil_to_tensor(original_image).to(device)
        inputs = processor(text=prompts, images=[original_image for _ in batch_questions], return_tensors="pt", padding=True).to(device)
        pixel_values = adv_processor.process(x_0)["pixel_values"]
        inputs['pixel_values'] = pixel_values
        """
        pass

    def process_batch(self, image_list: List[torch.Tensor]) -> dict:
        """
        Processes a batch of images into a format compatible with the LLaMA model.

        Args:
            image_list (List[torch.Tensor]): List of input image tensors, each with shape (C, H, W).

        Returns:
            dict: A dictionary with the following keys:
                - pixel_values (torch.Tensor): Batched processed image tensors, shape (batch_size, 1, max_image_tiles, channels, tile_height, tile_width).
                - aspect_ratio_ids: None (aspect ratio ids not calculated at this stage).
                - num_tiles: Real number of tiles for each image in the batch.
        """
        pass
