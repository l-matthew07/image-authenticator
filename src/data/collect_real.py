"""Collect real images from diverse sources."""

import os
import requests
import logging
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from PIL import Image
import io
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealImageCollector:
    """Collect real images from diverse sources to avoid dataset bias."""
    
    def __init__(self, output_dir: str = "data/raw/real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_coco(self, num_images: int = 1000, split: str = "val2017") -> List[str]:
        
        logger.info(f"Downloading {num_images} images from COCO {split}...")
        
        coco_url = f"http://images.cocodataset.org/{split}/"
        annotations_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        coco_dir = self.output_dir / "coco"
        coco_dir.mkdir(exist_ok=True)
        
        downloaded_paths = []
        
        try:
            # download sample images (val2017 has ~5000 images)
            # download a subset by ID range
            image_ids = range(397133, 397133 + min(num_images, 5000))
            
            for img_id in tqdm(image_ids[:num_images], desc="Downloading COCO images"):
                img_filename = f"{split}/{img_id:012d}.jpg"
                img_url = f"{coco_url}{img_id:012d}.jpg"
                img_path = coco_dir / f"{img_id:012d}.jpg"
                
                if img_path.exists():
                    downloaded_paths.append(str(img_path))
                    continue
                    
                try:
                    response = requests.get(img_url, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(io.BytesIO(response.content))
                        img.verify()
                        img_path.write_bytes(response.content)
                        downloaded_paths.append(str(img_path))
                except Exception as e:
                    logger.debug(f"Failed to download {img_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error downloading COCO images: {e}")
            
        logger.info(f"Downloaded {len(downloaded_paths)} images from COCO")
        return downloaded_paths
    
    
    def download_from_pexels(
        self, 
        num_images: int = 500,
        api_key: Optional[str] = None,
        query: str = "nature"
    ) -> List[str]:
        
        logger.info(f"Downloading {num_images} images from Pexels...")
        
        pexels_dir = self.output_dir / "pexels"
        pexels_dir.mkdir(exist_ok=True)
        
        downloaded_paths = []
        
        if not api_key:
            logger.warning(
                "No Pexels API key provided. "
                "For production use, get a free key from https://www.pexels.com/api/"
            )
            return downloaded_paths
        
        try:
            url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            params = {"query": query, "per_page": 80, "page": 1}
            
            downloaded = 0
            page = 1
            
            while downloaded < num_images:
                params["page"] = page
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code != 200:
                    logger.error(f"Pexels API error: {response.status_code}")
                    break
                
                data = response.json()
                photos = data.get("photos", [])
                
                if not photos:
                    break
                
                for photo in photos:
                    if downloaded >= num_images:
                        break
                    
                    img_url = photo["src"]["large"]
                    photo_id = photo["id"]
                    img_path = pexels_dir / f"{photo_id}.jpg"
                    
                    if img_path.exists():
                        downloaded_paths.append(str(img_path))
                        downloaded += 1
                        continue
                    
                    try:
                        img_response = requests.get(img_url, timeout=10)
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                            img.verify()
                            img_path.write_bytes(img_response.content)
                            downloaded_paths.append(str(img_path))
                            downloaded += 1
                    except Exception as e:
                        logger.debug(f"Failed to download image {photo_id}: {e}")
                        continue
                
                page += 1
                if page > data.get("total_results", 0) // 80:
                    break
                    
        except Exception as e:
            logger.error(f"Error downloading from Pexels: {e}")
        
        logger.info(f"Downloaded {len(downloaded_paths)} images from Pexels")
        return downloaded_paths
    
    def copy_from_local(
        self, 
        source_dir: str,
        num_images: Optional[int] = None,
        extensions: List[str] = [".jpg", ".jpeg", ".png"]
    ) -> List[str]:
       
        logger.info(f"Copying images from {source_dir}...")
        
        local_dir = self.output_dir / "local"
        local_dir.mkdir(exist_ok=True)
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory does not exist: {source_dir}")
            return []
        
        copied_paths = []
        image_files = []
        
        for ext in extensions:
            image_files.extend(source_path.rglob(f"*{ext}"))
            image_files.extend(source_path.rglob(f"*{ext.upper()}"))
        
        if num_images:
            image_files = image_files[:num_images]
        
        for img_file in tqdm(image_files, desc="Copying local images"):
            try:
                # Verify it's a valid image
                img = Image.open(img_file)
                img.verify()
                
                # Copy to output directory
                dest_path = local_dir / img_file.name
                # Handle name conflicts
                counter = 1
                while dest_path.exists():
                    stem = img_file.stem
                    dest_path = local_dir / f"{stem}_{counter}{img_file.suffix}"
                    counter += 1
                
                import shutil
                shutil.copy2(img_file, dest_path)
                copied_paths.append(str(dest_path))
            except Exception as e:
                logger.debug(f"Failed to copy {img_file}: {e}")
                continue
        
        logger.info(f"Copied {len(copied_paths)} images from local directory")
        return copied_paths
    
    def collect_all(
        self,
        num_images_per_source: int = 500,
        use_coco: bool = True,
        use_pexels: bool = False,
        pexels_api_key: Optional[str] = None,
        local_sources: Optional[List[str]] = None
    ) -> dict:
        
        all_paths = {}
        
        if use_coco:
            all_paths["coco"] = self.download_from_coco(num_images_per_source)
        
        if use_pexels and pexels_api_key:
            all_paths["pexels"] = self.download_from_pexels(
                num_images_per_source, 
                pexels_api_key
            )
        
        if local_sources:
            for local_source in local_sources:
                all_paths[f"local_{Path(local_source).name}"] = self.copy_from_local(
                    local_source,
                    num_images_per_source
                )
        
        # Save metadata
        metadata = {
            "total_images": sum(len(paths) for paths in all_paths.values()),
            "sources": {source: len(paths) for source, paths in all_paths.items()},
            "output_dir": str(self.output_dir)
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Collection complete: {metadata['total_images']} total images")
        return all_paths


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect real images from diverse sources")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/real",
        help="Output directory for collected images"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of images per source"
    )
    parser.add_argument(
        "--use-coco",
        action="store_true",
        default=True,
        help="Use COCO dataset"
    )
    parser.add_argument(
        "--use-pexels",
        action="store_true",
        help="Use Pexels API (requires --pexels-api-key)"
    )
    parser.add_argument(
        "--pexels-api-key",
        type=str,
        help="Pexels API key"
    )
    parser.add_argument(
        "--local-sources",
        type=str,
        nargs="+",
        help="Local directories to copy images from"
    )
    
    args = parser.parse_args()
    
    collector = RealImageCollector(output_dir=args.output_dir)
    collector.collect_all(
        num_images_per_source=args.num_images,
        use_coco=args.use_coco,
        use_pexels=args.use_pexels,
        pexels_api_key=args.pexels_api_key,
        local_sources=args.local_sources
    )


if __name__ == "__main__":
    main()
