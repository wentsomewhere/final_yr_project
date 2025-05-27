import os
import requests
from tqdm import tqdm
import argparse
import torch
from pathlib import Path

def download_file(url: str, destination: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_pretrained_models(models_dir: str):
    """Download pre-trained models."""
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs (replace with actual URLs when available)
    model_urls = {
        'srrgan': 'https://example.com/models/srrgan.pth',
        'ocr': 'https://example.com/models/ocr.pth',
    }
    
    for model_name, url in model_urls.items():
        model_path = os.path.join(models_dir, f'{model_name}.pth')
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            try:
                download_file(url, model_path)
                print(f"Successfully downloaded {model_name} model")
            except Exception as e:
                print(f"Failed to download {model_name} model: {str(e)}")
        else:
            print(f"{model_name} model already exists")

def verify_models(models_dir: str):
    """Verify downloaded models."""
    model_paths = {
        'srrgan': os.path.join(models_dir, 'srrgan.pth'),
        'ocr': os.path.join(models_dir, 'ocr.pth'),
    }
    
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            try:
                # Try loading the model
                checkpoint = torch.load(path, map_location='cpu')
                print(f"Successfully verified {model_name} model")
            except Exception as e:
                print(f"Failed to verify {model_name} model: {str(e)}")
        else:
            print(f"{model_name} model not found")

def main():
    parser = argparse.ArgumentParser(description='Download pre-trained models for SRR-GAN')
    parser.add_argument('--models_dir', type=str, default='models',
                      help='Directory to store downloaded models')
    args = parser.parse_args()
    
    # Download models
    download_pretrained_models(args.models_dir)
    
    # Verify models
    verify_models(args.models_dir)
    
    print("Model download and verification completed!")

if __name__ == '__main__':
    main() 