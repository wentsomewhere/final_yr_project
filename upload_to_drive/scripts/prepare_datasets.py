import os
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm
from pathlib import Path
import shutil
import logging
from typing import Optional
import torch
import torchvision
import numpy as np
from PIL import Image
import json

def setup_logging(log_dir: str = 'logs'):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'dataset_preparation.log')),
            logging.StreamHandler()
        ]
    )

def create_metadata(dataset_dir: str, text_type: str):
    """Create metadata file for a dataset."""
    metadata = {
        'text_type': text_type,
        'samples': []
    }
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            # Extract label from filename (format: idx_label.png)
            label = filename.split('_')[1].split('.')[0]
            metadata['samples'].append({
                'filename': filename,
                'text': label
            })
    
    # Save metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def download_file(url: str, destination: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def extract_archive(archive_path: str, extract_dir: str) -> bool:
    """Extract a zip or tar archive."""
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        return True
    except Exception as e:
        logging.error(f"Error extracting {archive_path}: {str(e)}")
        return False

def prepare_mnist_dataset(data_dir: str) -> bool:
    """Prepare MNIST dataset for handwritten text."""
    try:
        mnist_dir = os.path.join(data_dir, 'mnist')
        os.makedirs(mnist_dir, exist_ok=True)
        
        # Download MNIST dataset using torchvision
        train_dataset = torchvision.datasets.MNIST(
            root=mnist_dir,
            train=True,
            download=True
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=mnist_dir,
            train=False,
            download=True
        )
        
        # Create train/val/test splits
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Save splits
        for split, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            split_dir = os.path.join(mnist_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for idx, (image, label) in enumerate(dataset):
                # Convert tensor to numpy array
                if isinstance(image, torch.Tensor):
                    image_np = image.numpy()
                else:
                    image_np = np.array(image)
                
                # Convert to PIL Image
                image_pil = Image.fromarray(image_np)
                # Save image
                image_pil.save(os.path.join(split_dir, f'{idx:05d}_{label}.png'))
            
            # Create metadata for this split
            create_metadata(split_dir, 'handwritten')
        
        return True
    except Exception as e:
        logging.error(f"Error preparing MNIST dataset: {str(e)}")
        return False

def prepare_printed_text_dataset(data_dir: str) -> bool:
    """Prepare a synthetic printed text dataset."""
    try:
        printed_dir = os.path.join(data_dir, 'printed')
        os.makedirs(printed_dir, exist_ok=True)
        
        # Create synthetic text images
        from PIL import Image, ImageDraw, ImageFont
        import random
        import string
        
        def create_text_image(text, size=(256, 64), font_size=32):
            # Create a white background
            image = Image.new('L', size, color=255)
            draw = ImageDraw.Draw(image)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position to center it
            text_width = draw.textlength(text, font=font)
            text_height = font_size
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2
            
            # Draw the text
            draw.text((x, y), text, font=font, fill=0)
            return image
        
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(printed_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Generate different number of samples for each split
            n_samples = {'train': 1000, 'val': 200, 'test': 200}[split]
            
            for idx in range(n_samples):
                # Generate random text
                text_length = random.randint(5, 20)
                text = ''.join(random.choices(string.ascii_letters + string.digits, k=text_length))
                
                # Create and save image
                image = create_text_image(text)
                image.save(os.path.join(split_dir, f'{idx:05d}_{text}.png'))
            
            # Create metadata for this split
            create_metadata(split_dir, 'printed')
        
        return True
    except Exception as e:
        logging.error(f"Error preparing printed text dataset: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for SRR-GAN')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to store datasets')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to store logs')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Prepare datasets
    logging.info("Preparing MNIST Dataset for handwritten text...")
    if prepare_mnist_dataset(args.data_dir):
        logging.info("MNIST dataset preparation completed successfully")
    else:
        logging.error("MNIST dataset preparation failed")
    
    logging.info("Preparing Printed Text Dataset...")
    if prepare_printed_text_dataset(args.data_dir):
        logging.info("Printed text dataset preparation completed successfully")
    else:
        logging.error("Printed text dataset preparation failed")

if __name__ == '__main__':
    main() 