import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Dict, Optional, Union
import json
import cv2
from pathlib import Path

class TextImageDataset(Dataset):
    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True,
        text_type: str = 'both',
        min_text_length: int = 1,
        max_text_length: int = 100
    ):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.transform = transform or self._get_default_transform(is_training)
        self.is_training = is_training
        self.text_type = text_type
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # Load and merge metadata from all subfolders
        self.samples = self._gather_all_samples()
    
    def _gather_all_samples(self) -> List[dict]:
        samples = []
        for data_dir in self.data_dirs:
            data_dir = Path(data_dir)
            # Recursively find all metadata.json files
            for metadata_path in data_dir.rglob('metadata.json'):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Filter and adjust paths
                for item in metadata['samples']:
                    # Filter by text type
                    if self.text_type != 'both' and metadata.get('text_type', item.get('text_type', '')) != self.text_type:
                        continue
                    # Filter by text length
                    text_length = len(item['text'])
                    if text_length < self.min_text_length or text_length > self.max_text_length:
                        continue
                    # Adjust image path to be relative to the current metadata file
                    if 'filename' in item:
                        image_path = os.path.join(metadata_path.parent, item['filename'])
                    elif 'image_path' in item:
                        image_path = os.path.join(metadata_path.parent, item['image_path'])
                    else:
                        continue
                    samples.append({
                        'image_path': str(image_path),
                        'text': item['text']
                    })
        return samples
    
    def _get_default_transform(self, is_training: bool) -> transforms.Compose:
        if is_training:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, sample['text']

def create_dataloader(
    datasets: List[TextImageDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader from multiple datasets."""
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Create dataloader
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def create_metadata(
    data_dir: str,
    output_path: Optional[str] = None
) -> Dict:
    """Create metadata file for the dataset."""
    data_dir = Path(data_dir)
    metadata = {
        'char_set': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~',
        'samples': []
    }
    
    # Process handwritten text images
    handwritten_dir = data_dir / 'handwritten'
    if handwritten_dir.exists():
        for image_path in handwritten_dir.glob('*.png'):
            text = image_path.stem  # Assuming filename is the text
            metadata['samples'].append({
                'image_path': str(image_path.relative_to(data_dir)),
                'text': text,
                'text_type': 'handwritten',
                'quality': 'high'  # You might want to implement quality assessment
            })
    
    # Process printed text images
    printed_dir = data_dir / 'printed'
    if printed_dir.exists():
        for image_path in printed_dir.glob('*.png'):
            text = image_path.stem  # Assuming filename is the text
            metadata['samples'].append({
                'image_path': str(image_path.relative_to(data_dir)),
                'text': text,
                'text_type': 'printed',
                'quality': 'high'  # You might want to implement quality assessment
            })
    
    # Save metadata if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return metadata

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def get_albumentations_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    train_dataset = TextImageDataset(
        train_dir,
        transform=get_transforms(is_train=True),
        is_training=True
    )
    
    val_dataset = TextImageDataset(
        val_dir,
        transform=get_transforms(is_train=False),
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_low_res_image(image, scale_factor=4):
    """Create a low-resolution version of the input image."""
    h, w = image.shape[-2:]
    low_res_h, low_res_w = h // scale_factor, w // scale_factor
    
    # Downsample using bicubic interpolation
    low_res_image = transforms.Resize((low_res_h, low_res_w))(image)
    
    return low_res_image

class PairedTextImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, scale_factor=4, low_res_size=(64, 64), high_res_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_factor = scale_factor
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.dataset = TextImageDataset(root_dir, transform=None)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, text = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Explicitly convert to grayscale (1 channel)
        image = transforms.Grayscale(num_output_channels=1)(image)
        
        # Resize high-res image to fixed size
        high_res_image = transforms.Resize(self.high_res_size)(image)
        
        # Create low-res version and resize to fixed size
        low_res_image = create_low_res_image(high_res_image, self.scale_factor)
        low_res_image = transforms.Resize(self.low_res_size)(low_res_image)
        
        return low_res_image, high_res_image, text 