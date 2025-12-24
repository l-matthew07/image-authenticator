"""Collect AI-generated images from various generators."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
from PIL import Image
import json
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratedImageCollector:
    
    def __init__(self, output_dir: str = "data/raw/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def generate_with_stable_diffusion(
        self,
        num_images: int = 500,
        prompts: Optional[List[str]] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        save_metadata: bool = True
    ) -> List[Dict]:
        
        logger.info(f"Generating {num_images} images with Stable Diffusion...")
        
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            logger.error(
                "diffusers library not installed. Install with: pip install diffusers accelerate"
            )
            return []
        
        sd_dir = self.output_dir / "stable_diffusion"
        sd_dir.mkdir(exist_ok=True)
        
        if prompts is None:
            # Diverse prompts to avoid prompt bias
            prompts = [
                "a beautiful landscape with mountains and a lake",
                "a portrait of a person with a neutral expression",
                "a city street during daytime",
                "a close-up photograph of a flower",
                "an interior of a modern living room",
                "a group of people having a conversation",
                "a still life of fruits on a table",
                "a cat sitting on a windowsill",
                "a forest path in autumn",
                "a vintage car on a city street",
                "a beach scene with palm trees",
                "a professional headshot of a business person",
                "a kitchen with modern appliances",
                "a bird perched on a branch",
                "a urban skyline at sunset",
                "a detailed close-up of human hands",
                "a garden with various flowers",
                "a cozy coffee shop interior",
                "a dog playing in a park",
                "a mountain range with snow peaks"
            ]
        
        try:
            logger.info(f"Loading Stable Diffusion model: {model_id}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # disable for faster generation
                requires_safety_checker=False
            )
            pipe = pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            
            generated_data = []
            
            for i in tqdm(range(num_images), desc="Generating images"):
                prompt = prompts[i % len(prompts)]
                
                try:
                    with torch.inference_mode():
                        image = pipe(
                            prompt,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            height=512,
                            width=512
                        ).images[0]
                    
                    img_path = sd_dir / f"sd_{i:06d}.png"
                    image.save(img_path)
                    
                    metadata = {
                        "path": str(img_path),
                        "generator": "stable_diffusion",
                        "model_id": model_id,
                        "prompt": prompt,
                        "index": i
                    }
                    generated_data.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate image {i} with prompt '{prompt}': {e}")
                    continue
            
            if save_metadata:
                metadata_path = sd_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(generated_data, f, indent=2)
            
            logger.info(f"Generated {len(generated_data)} images with Stable Diffusion")
            return generated_data
            
        except Exception as e:
            logger.error(f"Error generating with Stable Diffusion: {e}")
            return []
    
    def copy_from_local_generated(
        self,
        source_dir: str,
        generator_name: str = "unknown",
        num_images: Optional[int] = None,
        extensions: List[str] = [".jpg", ".jpeg", ".png"]
    ) -> List[Dict]:
        
        logger.info(f"Copying generated images from {source_dir}...")
        
        gen_dir = self.output_dir / generator_name
        gen_dir.mkdir(exist_ok=True)
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return []
        
        copied_data = []
        image_files = []
        
        for ext in extensions:
            image_files.extend(source_path.rglob(f"*{ext}"))
            image_files.extend(source_path.rglob(f"*{ext.upper()}"))
        
        if num_images:
            image_files = image_files[:num_images]
        
        for idx, img_file in enumerate(tqdm(image_files, desc=f"Copying {generator_name} images")):
            try:
                img = Image.open(img_file)
                img.verify()
                
                dest_path = gen_dir / f"{generator_name}_{idx:06d}{img_file.suffix}"
                counter = 1
                while dest_path.exists():
                    stem = f"{generator_name}_{idx:06d}"
                    dest_path = gen_dir / f"{stem}_{counter}{img_file.suffix}"
                    counter += 1
                
                import shutil
                shutil.copy2(img_file, dest_path)
                
                metadata = {
                    "path": str(dest_path),
                    "generator": generator_name,
                    "original_path": str(img_file),
                    "index": idx
                }
                copied_data.append(metadata)
                
            except Exception as e:
                logger.debug(f"Failed to copy {img_file}: {e}")
                continue
        
        metadata_path = gen_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(copied_data, f, indent=2)
        
        logger.info(f"Copied {len(copied_data)} images from {generator_name}")
        return copied_data
    
    def generate_with_dalle_api(
        self,
        num_images: int = 100,
        prompts: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        model: str = "dall-e-3"
    ) -> List[Dict]:
       
        logger.info(f"Generating {num_images} images with DALL-E {model}...")
        
        # Try environment variable if api_key not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning(
                "No OpenAI API key provided. "
                "Get one from https://platform.openai.com/api-keys"
            )
            return []
        
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai library not installed. Install with: pip install openai")
            return []
        
        dalle_dir = self.output_dir / f"dalle_{model.replace('-', '_')}"
        dalle_dir.mkdir(exist_ok=True)
        
        if prompts is None:
            # Use similar diverse prompts as Stable Diffusion
            prompts = [
                "a beautiful landscape with mountains and a lake",
                "a portrait of a person with a neutral expression",
                "a city street during daytime",
                "a close-up photograph of a flower",
                "an interior of a modern living room"
            ]
        
        client = OpenAI(api_key=api_key)
        generated_data = []
        import requests
        from io import BytesIO
        
        for i in tqdm(range(num_images), desc="Generating DALL-E images"):
            prompt = prompts[i % len(prompts)]
            
            try:
                # Generate image
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size="1024x1024" if model == "dall-e-3" else "512x512",
                    quality="standard"
                )
                
                # Download image
                image_url = response.data[0].url
                img_response = requests.get(image_url, timeout=30)
                
                if img_response.status_code == 200:
                    # Save image
                    img_path = dalle_dir / f"dalle_{i:06d}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)
                    
                    metadata = {
                        "path": str(img_path),
                        "generator": f"dalle_{model.replace('-', '_')}",
                        "model": model,
                        "prompt": prompt,
                        "index": i
                    }
                    generated_data.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to generate DALL-E image {i}: {e}")
                continue
        
        # Save metadata
        metadata_path = dalle_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(generated_data, f, indent=2)
        
        logger.info(f"Generated {len(generated_data)} images with DALL-E")
        return generated_data
    
    def collect_all(
        self,
        num_images_per_generator: int = 500,
        use_stable_diffusion: bool = True,
        stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5",
        use_dalle: bool = False,
        dalle_api_key: Optional[str] = None,
        dalle_model: str = "dall-e-3",
        local_sources: Optional[Dict[str, str]] = None
    ) -> dict:
        """
        Collect images from all configured generators.
        
        Args:
            num_images_per_generator: Number of images per generator
            use_stable_diffusion: Whether to generate with Stable Diffusion
            stable_diffusion_model: Stable Diffusion model ID
            use_dalle: Whether to use DALL-E API
            dalle_api_key: OpenAI API key for DALL-E
            dalle_model: DALL-E model version
            local_sources: Dict mapping generator names to local directory paths
            
        Returns:
            Dictionary with generator names and lists of generated data
        """
        all_data = {}
        
        if use_stable_diffusion:
            all_data["stable_diffusion"] = self.generate_with_stable_diffusion(
                num_images=num_images_per_generator,
                model_id=stable_diffusion_model
            )
        
        if use_dalle and dalle_api_key:
            all_data[f"dalle_{dalle_model.replace('-', '_')}"] = self.generate_with_dalle_api(
                num_images=num_images_per_generator,
                api_key=dalle_api_key,
                model=dalle_model
            )
        
        if local_sources:
            for gen_name, local_path in local_sources.items():
                all_data[gen_name] = self.copy_from_local_generated(
                    source_dir=local_path,
                    generator_name=gen_name,
                    num_images=num_images_per_generator
                )
        
        # Save overall metadata
        metadata = {
            "total_images": sum(len(data) for data in all_data.values()),
            "generators": {gen: len(data) for gen, data in all_data.items()},
            "output_dir": str(self.output_dir)
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Collection complete: {metadata['total_images']} total generated images")
        return all_data


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect AI-generated images from various generators"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/generated",
        help="Output directory for collected images"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of images per generator"
    )
    parser.add_argument(
        "--use-stable-diffusion",
        action="store_true",
        default=True,
        help="Generate images with Stable Diffusion"
    )
    parser.add_argument(
        "--stable-diffusion-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID"
    )
    parser.add_argument(
        "--use-dalle",
        action="store_true",
        help="Generate images with DALL-E API (requires --dalle-api-key)"
    )
    parser.add_argument(
        "--dalle-api-key",
        type=str,
        help="OpenAI API key for DALL-E"
    )
    parser.add_argument(
        "--dalle-model",
        type=str,
        default="dall-e-3",
        choices=["dall-e-2", "dall-e-3"],
        help="DALL-E model version"
    )
    parser.add_argument(
        "--local-sources",
        type=str,
        nargs="+",
        help="Local directories to copy from (format: generator_name:path)"
    )
    
    args = parser.parse_args()
    
    # Parse local sources
    local_sources_dict = None
    if args.local_sources:
        local_sources_dict = {}
        for source in args.local_sources:
            if ":" in source:
                gen_name, path = source.split(":", 1)
                local_sources_dict[gen_name] = path
            else:
                logger.warning(f"Invalid local source format: {source} (expected 'name:path')")
    
    collector = GeneratedImageCollector(output_dir=args.output_dir)
    collector.collect_all(
        num_images_per_generator=args.num_images,
        use_stable_diffusion=args.use_stable_diffusion,
        stable_diffusion_model=args.stable_diffusion_model,
        use_dalle=args.use_dalle,
        dalle_api_key=args.dalle_api_key,
        dalle_model=args.dalle_model,
        local_sources=local_sources_dict
    )


if __name__ == "__main__":
    main()
