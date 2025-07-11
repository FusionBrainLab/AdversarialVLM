from typing import List, Tuple, Union
import PIL
import torch
import numpy as np
import torchvision
import random
from transformers.image_processing_utils import BatchFeature
from transformers import AutoProcessor, MllamaForConditionalGeneration, MllamaConfig, MllamaImageProcessor
from transformers.models.mllama.image_processing_mllama import get_optimal_tiled_canvas, get_image_size_fit_to_canvas, pack_images
import torch.nn.functional as F
from torchvision.transforms import functional as F_tv
from PIL import Image

def load_model_and_processor(model_name, device):
    """Load the model and processor."""
    model = MllamaForConditionalGeneration.from_pretrained(model_name).half().to(device)
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    return model, processor

def llama_backprocessing(processed_image: np.array, image_processor: MllamaImageProcessor, aspect_ratio_ids: int = 6):
    img_mean = image_processor.image_mean
    img_std = image_processor.image_std
    
    img_mean = np.reshape(img_mean, (3,1,1))
    img_std = np.reshape(img_std, (3,1,1))
    
    num_tiles, num_channels, height, width = processed_image.shape
    tile_height = image_processor.size["height"]
    tile_width = image_processor.size["width"]
    
    # supported_aspect_ratios = get_all_supported_aspect_ratios(processor.image_processor.max_image_tiles)
    # Supposed to be
    # [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]
    
    num_tiles_height, num_tiles_width = 2, 2
    if aspect_ratio_ids==5:
        processed_image = processed_image[[0,2,1,3]]
    # aspect_ratio = supported_aspect_ratios[aspect_ratio_ids-1] # height // tile_height
    
    # From (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    # To  (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    # To (num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)
    # TO (num_channels, num_tiles_height*tile_height, num_tiles_width*tile_width)
    image = processed_image.reshape(num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    image = image.transpose(2, 0, 3, 1, 4)
    glued_image = image.reshape(num_channels, num_tiles_height*tile_height, num_tiles_width*tile_width)
    
    # Denormalize
    glued_image = (((glued_image * img_std) + img_mean)*255).astype(np.uint8)
    
    # To channel last
    return glued_image.transpose(1, 2, 0)


def pad_to_max_num_crops_tensor(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if B < max_crops:
        pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images

class AdvMllamaInputs:
    refuses = [
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
        "Absolutely not."
    ]
    
    def __init__(self, questions, test_questions, batch_size, original_image, processor, device="cuda:0", target_text="sure, here it is!"):
        self.questions = questions
        self.test_questions = test_questions
        self.batch_size = batch_size
        self.processor = processor
        self.original_image = original_image
        self.device = device
        
        self.extra_token = "<|eot_id|>"
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
    
    def set_target_text(self, target_text: str):
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
        
    def update_refuse_tokens(self):
        self.refuse_tokens = self.processor.tokenizer(self.refuse_text+self.extra_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.refuse_suffix_length = self.refuse_tokens.shape[1]
        self.refuse = self.target_tokens[:, :-self.shift].repeat(self.batch_size, 1).to(self.device)

    def set_refuse(self, refuse_text: str):
        self.refuse_text = refuse_text
        self.update_refuse_tokens()
 
    def get_loss_refuse(self, logits):
        # Extract relevant logits and compute loss
        logits_suffix = logits[:, -self.refuse_suffix_length:-self.shift, :]
        logits_suffix = logits_suffix.permute(0, 2, 1)
        loss = F.cross_entropy(logits_suffix, self.refuse)
        return loss

    def get_inputs_refuse(self):
        batch_questions = random.choices(self.questions, k=self.batch_size)
        self.set_refuse(refuse = random.choice(self.refuses))
        
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
                        {"type": "text", "text": self.refuse_text} # TODO : Update this
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

class DifferentiableMllamaImageProcessor():
    def __init__(self, orig_processor, device):
        self.orig_processor = orig_processor
        self.device = device
        
        self.image_mean = torch.tensor(orig_processor.image_mean).view(-1, 1, 1).to(device)
        self.image_std = torch.tensor(orig_processor.image_std).view(-1, 1, 1).to(device)

        self.do_convert_rgb = orig_processor.do_convert_rgb
        self.do_rescale = False # not needed, tensors are already rescaled 
        self.do_normalize = orig_processor.do_normalize
        self.tile_size = orig_processor.size
        self.max_image_tiles = orig_processor.max_image_tiles
        self.rescale_factor = orig_processor.rescale_factor

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
            tensor_image, _ = self.resize_tensor(tensor_image)
        return tensor_image.to(self.device)

    def _optimal_size(self, image: torch.Tensor) -> Tuple[int, int]:
        # C x H x W
        num_channels, image_height, image_width = image.shape
        
        tile_size = self.tile_size["height"]
        max_image_tiles = self.max_image_tiles
        
        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=max_image_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size

        new_height, new_width = get_image_size_fit_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

        return new_height, new_width, (num_tiles_height, num_tiles_width)

    def resize_tensor(self, image: torch.Tensor) -> torch.Tensor:
        # C x H x W
        new_h, new_w, aspect_ratio = self._optimal_size(image)
        image = F.interpolate((image).unsqueeze(0), size=[int(new_h), int(new_w)], mode='bilinear', align_corners=False, antialias=True)
        image = image.squeeze(0)
        return image, aspect_ratio
    
    def pad(self, image: torch.Tensor, aspect_ratio: Tuple[int, int],):
        # C x H x W
        _, image_height, image_width = image.shape
        num_tiles_height, num_tiles_width = aspect_ratio
        padded_height = num_tiles_height * self.tile_size["height"]
        padded_width = num_tiles_width * self.tile_size["width"]
        
        # Calculate padding for each side (first for width, then for height)
        pad_left = 0
        pad_right = padded_width - image_width
        pad_top = 0
        pad_bottom = padded_height - image_height

        # Transform pad_size for torch.nn.functional.pad
        pad_size = [pad_left, pad_right, pad_top, pad_bottom]
        
        image = torch.nn.functional.pad(image.unsqueeze(0), pad_size, mode='constant', value=0.0)
        image = image.squeeze(0)
        return image

    def rescale(self, image: torch.Tensor):
        return image * self.rescale_factor
    
    def normalize(self, image: torch.Tensor):
        # C x H x W
        mean = self.image_mean.reshape(-1, 1, 1)
        std = self.image_std.reshape(-1, 1, 1)
        return (image - mean) / std

    def split_to_tiles(self, image: torch.Tensor, num_tiles_height: int, num_tiles_width: int):
        # C x H x W
        num_channels, height, width = image.shape
        tile_height = height // num_tiles_height
        tile_width = width // num_tiles_width
        
        if tile_height != self.tile_size["height"] or tile_width != self.tile_size["width"]:
            raise ValueError(f"Tile size must be {self.tile_size['height']}x{self.tile_size['width']}. Got: {tile_height}x{tile_width}")

        image = image.reshape(num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)

        # Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
        image = image.permute(1, 3, 0, 2, 4)

        # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
        image = image.reshape(num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

        return image
    
    def pack_images(self, batch_images: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        batch_size = len(batch_images)
        max_num_images = max([len(images) for images in batch_images])
        shapes = [image.shape for images in batch_images for image in images]
        _, channels, tile_height, tile_width = shapes[0]

        # Initialize the stacked images array with zeros
        stacked_images = torch.zeros(
            (batch_size, max_num_images, self.max_image_tiles, channels, tile_height, tile_width)
        ).to(self.device)

        # Fill the stacked images array with the tiled images from the batch
        all_num_tiles = []
        for i, images in enumerate(batch_images):
            num_sample_tiles = []
            for j, image in enumerate(images):
                num_tiles = image.shape[0]
                stacked_images[i, j, :num_tiles] = image
                num_sample_tiles.append(num_tiles)
            all_num_tiles.append(num_sample_tiles)

        return stacked_images, all_num_tiles
    
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
        """
        # C x H x W
        image, aspect_ratio = self.resize_tensor(image)

        image = self.pad(image, aspect_ratio=aspect_ratio)
        
        if self.do_rescale:
            image = self.rescale(image=image)

        if self.do_normalize:
            image = self.normalize(image=image)

        num_tiles_height, num_tiles_width = aspect_ratio
        image = self.split_to_tiles(image, num_tiles_height, num_tiles_width)
        # image: image_tiles, channels, tile_height, tile_width
        
        image_tiles, channels, tile_height, tile_width = image.shape
        image = image.reshape(1, 1, image_tiles, channels, tile_height, tile_width)
        
        image, _ = self.pack_images(image)
        
        # print(image.shape)
        
        # images: batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width
        # aspect_ratio_ids (np.ndarray) with shape (batch_size, max_num_images) - aspect ratio ids for each image, padded to max_num_images with 0
        # num_tiles (List[List[int]]) with (batch_size, num_images_in_batch) - real number of tiles for each image, not padded
        # aspect_ratio_mask (np.ndarray) with shape (batch_size, max_num_images, max_image_tiles) - number of tiles for each image, padded to max_num_images with 0
        data = {
            "pixel_values": image, 
            "aspect_ratio_ids": None,
            "num_tiles": image_tiles
        }
        return data
    
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
        # One image per batch is supposed
        data_list = []
        for image in image_list:
            data_list.append([self.process(image)["pixel_values"]])

        images, num_tiles = self.pack_images(data_list, self.max_image_tiles)

        return {
            "pixel_values": images,
            "aspect_ratio_ids": None,
            "num_tiles": num_tiles
        }

    def process_normalized(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def get_mask(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.ones_like(image)
        return self.process(image).to(dtype=torch.bool)

    def backprocessing_data(self, data: dict) -> torch.Tensor:
        """Back-processes the image to its original form."""
        image = data["pixel_values"][0]
        image = torch.nn.functional.interpolate(image.unsqueeze(0).float(), size=(data["image_sizes"]), mode='bicubic',).to(image.dtype)

        image = image * self.image_std + self.image_mean
        return image.detach()

    def tensor2pil(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x.clamp(0, 1)
        # img = llama_backprocessing(image[0][0], self.orig_processor)
        img = (x * 255).cpu().detach().permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        return img
