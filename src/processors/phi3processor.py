import random
from typing import Tuple, Union
import PIL
import torch
import numpy as np
import torchvision
from transformers.image_processing_utils import BatchFeature
from transformers import AutoProcessor, AutoModelForCausalLM
import torch.nn.functional as F
from torchvision.transforms import functional as F_tv
from PIL import Image

def pad_to_max_num_crops_tensor(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if B < max_crops:
        pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images


def load_model_and_processor(model_name, device):
    """Load the model and processor."""
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().to(device)
    processor = AutoProcessor.from_pretrained(model_name, num_crops=6, padding_side='left', trust_remote_code=True)
    return model, processor


class AdvPhiInputs:
    def __init__(self, questions, test_questions, batch_size, original_image, processor, device="cuda:0", target_text="sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.target_text = target_text
        self.original_image = original_image
        self.device = device
        
        extra_token = "<|end|>\n"
        self.target_tokens = processor.tokenizer(target_text+extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        self.shift = len(processor.tokenizer.encode(extra_token)) - 1 # first token is extra
        self.suffix_length = self.target_tokens.shape[1]#  +  3
        
        self.target = self.target_tokens[:, :-self.shift].repeat(batch_size, 1).to(self.device)
        # print("Suffix len:", self.suffix_length)
        # print("Shift len:", self.shift)
        # print("Target len:", self.target.shape)
        # self.target = inputs["input_ids"][:, -suffix_length:-shift]
    
    def get_loss(self, logits):
        # Extract relevant logits and compute loss
        logits_suffix = logits[:, -self.suffix_length:-self.shift, :]
        # print("Logits len:", logits_suffix.shape)
        logits_suffix = logits_suffix.permute(0, 2, 1)
        loss = F.cross_entropy(logits_suffix, self.target)
        return loss

    def get_inputs_train(self):
        batch_questions = random.choices(self.questions, k=self.batch_size)
        
        prompts = [f'<|user|>\n<|image_1|>\n{q}<|end|>\n<|assistant|>\n{self.target_text}<|end|>\n' for q in batch_questions]
        
        # inputs = self.processor(
        #     text=prompts,
        #     images=[self.original_image for _ in batch_questions],
        #     padding=True,
        #     return_tensors="pt",
        # ).to(torch.device(self.device))

        inputs = batch_processing(self.processor, prompts, [self.original_image]*self.batch_size).to(self.device)
        
        return inputs
        
    def get_inputs_inference(self, img, question = None):
        if question is not None:
            question = self.test_questions[0]

        inference_prompts = [f'<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n']
            
        inputs_for_inference = batch_processing(self.processor, inference_prompts, [img]).to(self.device)
        
        return inputs_for_inference

class DifferentiablePhi3VImageProcessor():
    def __init__(self, orig_processor, device):
        """
        orig_processor : "Phi3VImageProcessor"
        
        """
        self.image_mean = torch.tensor(orig_processor.image_mean).view(-1, 1, 1).to(device)
        self.image_std = torch.tensor(orig_processor.image_std).view(-1, 1, 1).to(device)
        self.do_convert_rgb = orig_processor.do_convert_rgb
        self.num_crops = orig_processor.num_crops
        self.num_img_tokens = orig_processor.num_img_tokens
    
    
    def fit_size_pil(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Resize a PIL image so that it has the optimal size for phi3 model.
        Args:
            image: PIL Image
        
        Returns:
            resized PIL Image
        """
        new_h, new_w = self._optimal_size(image, self.num_crops)
        image = image.resize((new_h, new_w), resample=PIL.Image.BILINEAR)
        return image
    
    def fit_size_tensor(self, image: torch.Tensor) -> torch.Tensor:
        new_h, new_w = self._optimal_size(image, self.num_crops)
        image = F.interpolate(image.unsqueeze(0), size=[new_h, new_w], mode='bilinear', align_corners=False)
        image = image.squeeze(0)
        return image
    
    def pil_to_tensor(self, image: PIL.Image, resize: bool = False) -> torch.Tensor:
        image = image.convert("RGB")
        if resize:
            image = self.fit_size_pil(image)
        return torch.tensor(np.array(image).astype(np.float32) / 255).permute(2, 0, 1)

    def _optimal_size(self, image: Union[torch.Tensor, PIL.Image], hd_num: int) -> Tuple[int, int]:
        if type(image) == torch.Tensor:
            _, height, width = image.shape
        else:
            height, width = image.size
        trans = False

        # Transpose if width is less than height
        if width < height:
            trans = True
            height, width = width, height

        # Calculate aspect ratio and scale
        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= hd_num:
            scale += 1
        scale -= 1

        # Calculate new dimensions
        new_w = int(scale * 336)
        new_h = int(new_w / ratio)
        
        if trans:
            return new_w, new_h

        return new_h, new_w

    def _pad(self, img: torch.Tensor, hd_num=16):
        # Get height and width from tensor dimensions (C, H, W)
        _, height, width = img.shape
        trans = False

        # Transpose if width is less than height to ensure larger size is divisible by 336
        if width < height:
            img = img.transpose(2, 1)  # Swap height and width in tensor (C, H, W)
            trans = True
            _, height, width = img.shape
        
        # Calculate aspect ratio and scale
        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= hd_num:
            scale += 1
        scale -= 1

        # Calculate new dimensions
        new_w = int(scale * 336)
        new_h = int(new_w / ratio)
        img = torch.nn.functional.interpolate(img.unsqueeze(0).float(), size=[new_h, new_w], mode='bilinear',).to(img.dtype)
        img = img.squeeze(0)
        
        height, width = img.shape[1], img.shape[2]

        # Check if larger dimension (width) is divisible by 336 and max patches <= hd_num
        if width % 336 != 0 or (width // 336) * (height // 336) > hd_num:
            raise ValueError("Max side length must be divisible by 336, and max patches must be <= hd_num.")
        
        # Calculate padding for height to make it divisible by 336
        target_h = int(np.ceil(height / 336) * 336)
        pad_top = (target_h - height) // 2
        pad_bottom = target_h - height - pad_top

        # Apply padding using F.pad (differentiable)
        img = torch.nn.functional.pad(img.unsqueeze(0), [0, 0, pad_top, pad_bottom], mode='constant', value=1.0)
        img = img.squeeze(0)

        # Transpose back if needed
        if trans:
            img = img.transpose(2, 1)

        return img
    
    def _process(self, image: torch.Tensor) -> torch.Tensor:
        # create global image 
        global_image = torch.nn.functional.interpolate(image.unsqueeze(0).float(), size=(336, 336), mode='bicubic',).to(image.dtype)

        # [(3, h, w)], where h, w is multiple of 336
        _, h, w = image.shape

        # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
        # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
        hd_image_reshape = image.reshape(1, 3, h//336, 336, w//336, 336).permute(0,2,4,1,3,5).reshape(-1, 3, 336, 336).contiguous()
        # concat global image and local image
        image_transformed = torch.cat([global_image] + [hd_image_reshape], dim=0)
        
        # pad to max_num_crops
        B, _, H, W = image_transformed.shape
        if B < self.num_crops+1:
            pad = torch.zeros(self.num_crops+1 - B, 3, H, W, dtype=image_transformed.dtype, device=image_transformed.device)
            image_transformed = torch.cat([image_transformed, pad], dim=0)

        return image_transformed

    def process(self, image: torch.Tensor) -> dict:        
        image = self._pad(image, hd_num = self.num_crops)
        image = (image - self.image_mean) / self.image_std
        image_transformed = self._process(image)
        _, h, w = image.shape
        num_img_tokens = int(((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12)
        data = {
            "pixel_values": image_transformed.unsqueeze(0),  # (1, TILES, CHANNELS, H, W)
            "image_sizes": [[h, w]],
            "num_img_tokens": [num_img_tokens]
        }
        return data
    
    def process_normalized(self, image: torch.Tensor) -> torch.Tensor:
        image = self._pad(image, hd_num = self.num_crops)
        return self._process(image)
    
    def get_mask(self, image: torch.Tensor) -> torch.Tensor:        
        image = torch.ones_like(image)
        return self._process(image).to(dtype=torch.bool)
    
    def backprocessing_data(self, data: dict) -> torch.Tensor:
        """Back-processes the image to its original form."""
        image = data["pixel_values"][0]
        image = torch.nn.functional.interpolate(image.unsqueeze(0).float(), size=(data["image_sizes"]), mode='bicubic',).to(image.dtype)
        
        image = image * self.image_std + self.image_mean
        return image.detach()
    
    def tensor2pil(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x.clamp(0, 1)
        img = (x * 255).cpu().detach().permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img


def batch_processing(processor, batch, images):
    listof_inputs: list[BatchFeature] = []
    for prompt, image in zip(batch, images):
        if not isinstance(prompt, str):
            prompt = processor.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
        listof_inputs.append(inputs)
    
    inputs = stack_and_pad_inputs(
        listof_inputs, pad_token_id=processor.tokenizer.pad_token_id
    )
    
    return inputs


def stack_and_pad_inputs(inputs: list[BatchFeature], pad_token_id: int) -> BatchFeature:
    listof_input_ids = [i.input_ids[0] for i in inputs]
    new_input_ids = pad_left(listof_input_ids, pad_token_id=pad_token_id)
    data = dict(
        pixel_values=torch.cat([i.pixel_values for i in inputs], dim=0),
        image_sizes=torch.cat([i.image_sizes for i in inputs], dim=0),
        input_ids=new_input_ids,
        attention_mask=(new_input_ids != pad_token_id).long(),
    )
    new_inputs = BatchFeature(data).to("cuda")
    return new_inputs


def pad_left(seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """Example: pad_left([[1, 2], [3, 4, 5]], pad_token_id=0) -> [[0, 1, 2], [3, 4, 5]]"""
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq) :] = seq
    return padded