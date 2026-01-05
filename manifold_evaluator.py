"""
Refactored Diffusion Model Evaluation Framework
Optimized for large-scale testing with better modularity and memory efficiency
Uses CLIP-based criterion from original implementation
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
import torchvision

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPModel, AutoImageProcessor, pipeline as pipeline_caption
import torchvision.transforms.functional as TF


# ==================== Configuration ====================

@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    # Model settings
    model_name: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Criterion settings
    num_noise: int = 8
    time_frac: float = 0.01
    epsilon_reg: float = 1e-8
    image_size: int = 512
    
    # Batch processing
    batch_size: int = 4
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "results"
    save_intermediates: bool = False
    return_terms: bool = False
    
    # Caption settings
    use_provided_prompts: bool = False
    caption_model: str = "llava-hf/llava-1.5-7b-hf"
    
    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ==================== Logging Setup ====================

def setup_logging(output_dir: str, level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==================== Model Manager ====================

class ModelManager:
    """Manages model loading and memory"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.float32
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Models (lazy loading)
        self._pipeline = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
        self._clip = None
        self._processor = None
        self._image_to_text = None
        
    def load_sd_pipeline(self):
        """Load Stable Diffusion pipeline"""
        if self._pipeline is None:
            self.logger.info(f"Loading SD pipeline: {self.config.model_name}")
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype
            )
            self._pipeline.to(self.device)
            
            # Extract components
            self._unet = self._pipeline.unet.eval()
            self._vae = self._pipeline.vae.eval()
            self._text_encoder = self._pipeline.text_encoder.eval()
            self._tokenizer = self._pipeline.tokenizer
            self._scheduler = DDPMScheduler.from_pretrained(
                self.config.model_name,
                subfolder="scheduler"
            )
            
            self.logger.info("SD pipeline loaded successfully")
        
        return (self._unet, self._tokenizer, self._text_encoder, 
                self._vae, self._scheduler)
    
    def load_clip(self):
        """Load CLIP model"""
        if self._clip is None:
            self.logger.info("Loading CLIP model")
            self._clip = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            self._processor = AutoImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.logger.info("CLIP loaded successfully")
        
        return self._clip, self._processor
    
    def load_caption_model(self):
        """Load image captioning model"""
        if self._image_to_text is None:
            self.logger.info(f"Loading caption model: {self.config.caption_model}")
            self._image_to_text = pipeline_caption(
                "image-text-to-text",
                model=self.config.caption_model,
                device=self.device
            )
            self.logger.info("Caption model loaded successfully")
        
        return self._image_to_text
    
    def clear_cache(self):
        """Clear GPU cache"""
        torch.cuda.empty_cache()
        self.logger.info("GPU cache cleared")
    
    def unload_all(self):
        """Unload all models to free memory"""
        self._pipeline = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
        self._clip = None
        self._processor = None
        self._image_to_text = None
        self.clear_cache()
        self.logger.info("All models unloaded")


# ==================== Image Processing ====================

class ImageProcessor:
    """Handles image loading and preprocessing"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.float32
        
    @staticmethod
    def resize_and_crop(img_t: torch.Tensor, size: int) -> torch.Tensor:
        """Resize and center crop image tensor"""
        import torchvision
        
        # Resize slightly larger - IMPORTANT: use the same method as original
        img_t = torchvision.transforms.Resize(size + 3)(img_t)
        
        # Center crop
        start_x = (img_t.size(-1) - size) // 2
        start_y = (img_t.size(-2) - size) // 2
        
        if img_t.dim() == 3:  # CHW
            return img_t[:, start_y:start_y + size, start_x:start_x + size]
        elif img_t.dim() == 4:  # BCHW
            return img_t[:, :, start_y:start_y + size, start_x:start_x + size]
        else:
            raise ValueError(f"Unsupported tensor shape: {img_t.shape}")
    
    def load_image(self, path: str) -> torch.Tensor:
        """Load single image from path"""
        img = imageio.imread(path, pilmode='RGB')
        return torch.tensor(img, device=self.device)
    
    def load_image_batch(self, paths: List[str]) -> List[torch.Tensor]:
        """Load batch of images from paths"""
        return [self.load_image(p) for p in paths]
    
    def preprocess(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Preprocess images to [-1, 1] range BCHW format"""
        # Convert HWC to CHW and resize
        processed = []
        for img in images:
            # First convert to float and normalize, THEN convert dtype
            img = img.float().permute(2, 0, 1)  # HWC -> CHW, keep as float32
            img = self.resize_and_crop(img, self.config.image_size)
            processed.append(img)
        
        # Stack and normalize to [-1, 1]
        batch = torch.stack(processed)
        batch = 2.0 * (batch / 255.0) - 1.0
        
        # Only convert to half precision after normalization
        if self.dtype == torch.float16:
            batch = batch.half()
        
        return batch
    
    def postprocess(self, img_t: torch.Tensor) -> np.ndarray:
        """Convert from [-1, 1] BCHW to [0, 255] BHWC uint8"""
        img_t = (img_t / 2.0 + 0.5).clamp(0, 1) * 255
        img_t = img_t.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return img_t.round().astype(np.uint8)


# ==================== Caption Generator ====================

class CaptionGenerator:
    """Generates captions for images"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_caption(self, image_tensor: torch.Tensor) -> str:
        """Generate caption for single image"""
        image_to_text = self.model_manager.load_caption_model()
        
        # Convert tensor to PIL
        image_pil = TF.to_pil_image(image_tensor.permute(2, 0, 1))
        
        prompt = "Generate a caption for the image that contains only facts and detailed."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt},
            ],
        }]
        
        output = image_to_text(text=messages, generate_kwargs={"max_new_tokens": 76})
        caption = output[0]["generated_text"][-1]["content"].strip()
        
        return caption
    
    def generate_captions_batch(self, images: List[torch.Tensor]) -> List[str]:
        """Generate captions for batch of images"""
        captions = []
        for img in tqdm(images, desc="Generating captions"):
            captions.append(self.generate_caption(img))
        return captions


# ==================== CLIP Criterion ====================

class CLIPCriterion:
    """Implements CLIP-based criterion (original method)"""
    
    def __init__(self, config: EvalConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
        
        # CLIP dimension
        self.d_clip = 512
        self.sqrt_d_clip = self.d_clip ** 0.5
    
    @staticmethod
    def normalize_batch(batch: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Normalize each element in batch"""
        dims = tuple(range(1, batch.dim()))
        norms = torch.norm(batch, p=2, dim=dims, keepdim=True)
        return batch / (norms + epsilon)
    
    @staticmethod
    def numpy_chunk(arr: np.ndarray, num_chunks: int) -> List[np.ndarray]:
        """Split numpy array into chunks"""
        chunk_size = arr.shape[0] // num_chunks
        remainder = arr.shape[0] % num_chunks
        indices = np.cumsum([0] + [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)])
        return np.split(arr, indices[1:-1], axis=0)
    
    def compute_text_embeddings(
        self,
        prompts: List[str],
        num_noise: int
    ) -> torch.Tensor:
        """Compute text embeddings for prompts"""
        _, tokenizer, text_encoder, _, _ = self.model_manager.load_sd_pipeline()
        
        # Expand prompts for num_noise
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_noise)
        
        # Tokenize
        text_tokens = tokenizer(
            expanded_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = text_tokens.input_ids.to(self.device)
        
        with torch.no_grad():
            text_emb = text_encoder(input_ids).last_hidden_state
        
        return text_emb
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space"""
        _, _, _, vae, _ = self.model_manager.load_sd_pipeline()
        
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()
            z0 = z0 * vae.config.scaling_factor
        
        return z0
    
    def decode_latents(self, latents: torch.Tensor, sub_batch_size: int = 16) -> torch.Tensor:
        """Decode latents to images with batch splitting"""
        _, _, _, vae, _ = self.model_manager.load_sd_pipeline()
        
        # IMPORTANT: Scale back BEFORE decoding (like original code)
        latents = 1.0 / vae.config.scaling_factor * latents
        
        if latents.size(0) <= sub_batch_size:
            with torch.no_grad():
                decoded = vae.decode(latents, return_dict=False)[0]
        else:
            # Split into sub-batches to avoid OOM
            num_sub_batches = (latents.size(0) + sub_batch_size - 1) // sub_batch_size
            self.logger.info(f"Decoding {latents.size(0)} latents in {num_sub_batches} sub-batches")
            
            decoded_list = []
            with torch.no_grad():
                for i in range(num_sub_batches):
                    start_idx = i * sub_batch_size
                    end_idx = min((i + 1) * sub_batch_size, latents.size(0))
                    
                    torch.cuda.empty_cache()
                    decoded_sub = vae.decode(latents[start_idx:end_idx], return_dict=False)[0]
                    decoded_list.append(decoded_sub)
                    del decoded_sub
            
            decoded = torch.cat(decoded_list, dim=0)
        
        return decoded
    
    def postprocess_decoded(self, img_t: torch.Tensor, siz: int, do_resize: bool = True) -> np.ndarray:
        """Convert decoded images to numpy uint8 format - EXACTLY like original"""
        if do_resize:
            # Apply resize like original code
            import torchvision
            img_t = torchvision.transforms.Resize(siz + 3)(img_t)
            start_x = (img_t.size(-1) - siz) // 2
            start_y = (img_t.size(-2) - siz) // 2
            img_t = img_t[:, :, start_y:start_y + siz, start_x:start_x + siz]
        
        # Scale exactly like original: (img_t / 2 + 0.5).clamp(0, 1) * 255
        img_t = (img_t / 2.0 + 0.5).clamp(0, 1) * 255
        img_t = img_t.detach().cpu()
        img_t = img_t.permute(0, 2, 3, 1).float().numpy()
        return img_t.round().astype(np.uint8)
    
    def compute_clip_features(self, images) -> torch.Tensor:
        """Compute CLIP features for images"""
        clip, processor = self.model_manager.load_clip()
        
        with torch.no_grad():
            # Handle both numpy arrays and tensors
            if isinstance(images, np.ndarray):
                inputs = processor(images=images, return_tensors="pt").to(self.device)
            elif isinstance(images, torch.Tensor):
                # Convert tensor to numpy for processor
                if images.dim() == 3:  # Single image HWC
                    images_np = images.cpu().numpy()
                else:  # Batch
                    images_np = images.cpu().numpy()
                inputs = processor(images=images_np, return_tensors="pt").to(self.device)
            else:
                inputs = processor(images=images, return_tensors="pt").to(self.device)
            
            features = clip.get_image_features(**inputs).detach().cpu()
        
        return features
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: List[torch.Tensor]
    ) -> List[Dict]:
        """Evaluate a batch of images using CLIP-based criterion"""
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Load models
        unet, _, _, vae, scheduler = self.model_manager.load_sd_pipeline()
        
        # Get text embeddings
        text_emb = self.compute_text_embeddings(prompts, num_noise)
        
        # Encode images to latents
        latents = self.encode_images(images)
        latents = latents.repeat_interleave(num_noise, dim=0).half()
        
        # Sample spherical noise
        gauss_noise = torch.randn_like(latents, device=self.device, dtype=torch.float16)
        spherical_noise = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        
        # Scale noise by sqrt(dimension)
        sqrt_d = torch.prod(torch.tensor(latents.shape[1:], device=self.device)).float().sqrt()
        spherical_noise = spherical_noise * sqrt_d
        
        # Get timestep
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        
        # Log alpha_t and dimension info
        alpha_t = scheduler.alphas_cumprod[timestep[0].item()]
        self.logger.info(f"Timestep: {timestep[0].item()}, alpha_t: {alpha_t.item():.6f}")
        self.logger.info(f"Latent dimension: {torch.prod(torch.tensor(latents.shape[1:])).item()}")
        
        # Add noise to latents
        noisy_latents = scheduler.add_noise(
            original_samples=latents,
            noise=spherical_noise,
            timesteps=timestep
        ).half()
        
        # Predict noise with UNet
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        
        # Clean up intermediate tensors
        del noisy_latents, timestep, gauss_noise, text_emb, images
        
        # Decode noise predictions and spherical noise
        decoded_noise = self.decode_latents(noise_pred)
        decoded_spherical_noise = self.decode_latents(spherical_noise)
        
        # Convert to numpy
        # decoded_noise_np = self.postprocess_decoded(decoded_noise)
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_spherical_noise_np = self.postprocess_decoded(decoded_spherical_noise, siz, do_resize=True)
        # decoded_spherical_noise_np = self.postprocess_decoded(decoded_spherical_noise)
        
        # Split into per-image chunks
        decoded_noise_chunks = self.numpy_chunk(decoded_noise_np, num_images)
        decoded_spherical_chunks = self.numpy_chunk(decoded_spherical_noise_np, num_images)
        
        # Compute CLIP-based criterion for each image
        results = []
        for i, (noise_chunk, spherical_chunk, img_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_spherical_chunks, images_raw)
        ):
            # Get CLIP features for original image
            img_s = img_raw.float().cpu().numpy()
            img_clip = self.compute_clip_features(img_s)
            
            # Get CLIP features for decoded noise
            img_d_clip = self.compute_clip_features(noise_chunk)
            
            # Get CLIP features for decoded spherical noise
            img_s_clip = self.compute_clip_features(spherical_chunk)
            
            # Compute similarity metrics
            bias_vec = self.cos(img_clip, img_d_clip).numpy()
            kappa_vec = self.cos(img_d_clip, img_s_clip).numpy()
            D_vec = torch.norm(
                img_d_clip.view(img_d_clip.size(0), -1),
                p=2,
                dim=1
            ).cpu().numpy()
            
            # Aggregate over noise samples
            bias_mean = bias_vec.mean()
            kappa_mean = kappa_vec.mean()
            D_mean = D_vec.mean()
            
            # Compute criterion using coefficients from paper
            criterion = 1 + (self.sqrt_d_clip * bias_mean - D_mean + kappa_mean) / (self.sqrt_d_clip + 2)
            
            result = {"criterion": float(criterion)}
            
            if self.config.return_terms:
                result.update({
                    "bias": float(bias_mean),
                    "kappa": float(kappa_mean),
                    "D": float(D_mean),
                })
            
            results.append(result)
        
        return results


# ==================== Evaluator ====================

class DiffusionEvaluator:
    """Main evaluator class"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        self.logger.info(f"Initialized evaluator with config: {config}")
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.image_processor = ImageProcessor(config)
        self.caption_generator = CaptionGenerator(self.model_manager)
        self.criterion = CLIPCriterion(config, self.model_manager)
        
        # Save config
        config.save(os.path.join(config.output_dir, "config.json"))
    
    def evaluate_images(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """Evaluate a list of images"""
        self.logger.info(f"Evaluating {len(image_paths)} images")
        
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), self.config.batch_size), 
                     desc="Processing batches"):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            # Load and preprocess images
            raw_images = self.image_processor.load_image_batch(batch_paths)
            processed_images = self.image_processor.preprocess(raw_images)
            
            # Get or generate prompts
            if prompts is not None:
                batch_prompts = prompts[i:i + self.config.batch_size]
            else:
                batch_prompts = self.caption_generator.generate_captions_batch(raw_images)
            
            # Evaluate - pass raw images for CLIP computation
            batch_results = self.criterion.evaluate_batch(
                processed_images,
                batch_prompts,
                raw_images  # Pass raw images for CLIP
            )
            
            # Add metadata
            for j, result in enumerate(batch_results):
                result["image_path"] = batch_paths[j]
                result["prompt"] = batch_prompts[j]
            
            all_results.extend(batch_results)
            
            # Clear cache periodically
            if (i // self.config.batch_size) % 10 == 0:
                self.model_manager.clear_cache()
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = "results.json"):
        """Save results to JSON"""
        output_path = os.path.join(self.config.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        criteria = [r["criterion"] for r in results]
        
        self.logger.info("="*50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total images: {len(results)}")
        self.logger.info(f"Mean criterion: {np.mean(criteria):.6f}")
        self.logger.info(f"Std criterion: {np.std(criteria):.6f}")
        self.logger.info(f"Min criterion: {np.min(criteria):.6f}")
        self.logger.info(f"Max criterion: {np.max(criteria):.6f}")
        self.logger.info("="*50)


# ==================== Main Entry Point ====================

def main():
    """Example usage"""
    # Create configuration
    config = EvalConfig(
        device="cuda",
        batch_size=2,
        num_noise=8,
        time_frac=0.01,
        image_size=512,
        output_dir="results/experiment_1",
        return_terms=True,
    )
    
    # Initialize evaluator
    evaluator = DiffusionEvaluator(config)
    
    # Example image paths
    image_paths = [
        "example_images/real_car.jpg",
        "example_images/real_dog.png",
        "example_images/gen_okapi.png",
        "example_images/gen_frog.png",
    ]
    
    # Optional: provide prompts
    prompts = None  # Set to list of strings to use custom prompts
    
    # Evaluate
    results = evaluator.evaluate_images(image_paths, prompts)
    
    # Save and display results
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    # Print individual results
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Criterion: {result['criterion']:.6f}")


if __name__ == "__main__":
    main()