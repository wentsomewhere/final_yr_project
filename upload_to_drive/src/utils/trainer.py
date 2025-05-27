import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
import psutil
import time
from src.models.srrgan import Generator, Discriminator, VGGFeatureExtractor
from src.models.ocr import OCRModule
from src.utils.metrics import calculate_psnr, calculate_ssim
import torch.nn.functional as F

class ResourceAwareTrainer:
    def __init__(self, base_batch_size: int = 8):
        self.base_batch_size = base_batch_size
        self.current_batch_size = base_batch_size
        self.min_batch_size = 2
        self.max_batch_size = base_batch_size
        self.resource_check_interval = 10  # Check resources every 10 batches
        self.batch_count = 0
        
    def check_resources(self) -> bool:
        """Check if system resources are within limits."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # If any resource is above 85%, return False
        if memory.percent > 85 or cpu_percent > 85:
            return False
        return True
    
    def adjust_batch_size(self, current_batch_size: int) -> int:
        """Adjust batch size based on resource usage."""
        if not self.check_resources():
            # Reduce batch size by half, but not below minimum
            new_batch_size = max(current_batch_size // 2, self.min_batch_size)
            logging.warning(f"High resource usage detected. Reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        elif current_batch_size < self.max_batch_size:
            # Try to increase batch size if resources are available
            new_batch_size = min(current_batch_size * 2, self.max_batch_size)
            if self.check_resources():
                logging.info(f"Resources available. Increasing batch size from {current_batch_size} to {new_batch_size}")
                return new_batch_size
        return current_batch_size

class Trainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        ocr_model: OCRModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        lambda_perceptual: float = 1.0,
        lambda_ocr: float = 1.0,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.ocr_model = ocr_model
        self.feature_extractor = VGGFeatureExtractor().to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize resource-aware training
        self.resource_manager = ResourceAwareTrainer(base_batch_size=train_loader.batch_size)
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ocr = lambda_ocr
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize best PSNR
        self.best_psnr = 0.0
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics['psnr'])
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with resource-aware batch size adjustment."""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_psnr = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            try:
                # Check and adjust batch size periodically
                if batch_idx % self.resource_manager.resource_check_interval == 0:
                    new_batch_size = self.resource_manager.adjust_batch_size(self.train_loader.batch_size)
                    if new_batch_size != self.train_loader.batch_size:
                        # Recreate data loader with new batch size
                        self.train_loader = DataLoader(
                            self.train_loader.dataset,
                            batch_size=new_batch_size,
                            shuffle=True,
                            num_workers=2
                        )
                        pbar = tqdm(self.train_loader, desc='Training')
                
                # Get data
                lr_imgs, hr_imgs, _ = batch
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Train discriminator
                d_loss = self._train_discriminator(lr_imgs, hr_imgs)
                
                # Train generator
                g_loss = self._train_generator(lr_imgs, hr_imgs)
                
                # Calculate metrics
                with torch.no_grad():
                    sr_imgs = self.generator(lr_imgs)
                    psnr = calculate_psnr(sr_imgs, hr_imgs)
                
                # Update progress
                total_g_loss += g_loss
                total_d_loss += d_loss
                total_psnr += psnr
                num_batches += 1
                
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}',
                    'D_Loss': f'{d_loss:.4f}',
                    'PSNR': f'{psnr:.2f}',
                    'Batch Size': self.train_loader.batch_size
                })
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                self._save_emergency_checkpoint(batch_idx)
                continue
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches,
            'psnr': total_psnr / num_batches
        }
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.generator.eval()
        
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Get data
                lr_imgs, hr_imgs, _ = batch  # Unpack the tuple
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Generate super-resolution images
                sr_imgs = self.generator(lr_imgs)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                ssim = calculate_ssim(sr_imgs, hr_imgs)
                
                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1
        
        return {
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }
    
    def _train_discriminator(self, lr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """Train discriminator for one batch."""
        self.d_optimizer.zero_grad()
        real_preds = self.discriminator(hr_imgs)
        fake_imgs = self.generator(lr_imgs)
        fake_preds = self.discriminator(fake_imgs.detach())
        d_loss_real = self.bce_loss(real_preds, torch.ones_like(real_preds))
        d_loss_fake = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        return d_loss.item()
    
    def _train_generator(self, lr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """Train generator for one batch."""
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        fake_imgs = self.generator(lr_imgs)
        
        # Adversarial loss
        fake_preds = self.discriminator(fake_imgs)
        g_loss_adv = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
        
        # Reconstruction loss
        g_loss_rec = self.l1_loss(fake_imgs, hr_imgs)
        
        # Perceptual loss
        g_loss_perceptual = self._calculate_perceptual_loss(fake_imgs, hr_imgs)
        
        # OCR loss
        g_loss_ocr = self._calculate_ocr_loss(fake_imgs, hr_imgs)
        
        # Total loss
        g_loss = (
            g_loss_adv +
            g_loss_rec +
            self.lambda_perceptual * g_loss_perceptual +
            self.lambda_ocr * g_loss_ocr
        )
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def _calculate_perceptual_loss(self, fake_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss using VGG features."""
        # Extract features using VGG feature extractor
        fake_features = self.feature_extractor(fake_imgs)
        hr_features = self.feature_extractor(hr_imgs)
        
        # Calculate loss
        loss = self.l1_loss(fake_features, hr_features)
        return loss
    
    def _calculate_ocr_loss(self, fake_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> torch.Tensor:
        """Calculate OCR loss between fake and real images."""
        try:
            # Convert grayscale to RGB if needed
            if fake_imgs.size(1) == 1:
                fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
            if hr_imgs.size(1) == 1:
                hr_imgs = hr_imgs.repeat(1, 3, 1, 1)
            
            # Calculate dimensions that will work with the CRNN architecture
            # The CNN layers reduce height by factor of 16 (4 pooling layers)
            # We want final height to be 1, so input height should be 32
            target_height = 32
            
            # Calculate width that will work with the RNN layers
            # The CNN layers reduce width by factor of 4 (2 pooling layers)
            # We want final width to be around 32 for the RNN
            # So input width should be 32 * 4 = 128
            target_width = 128
            
            # Resize images to these dimensions
            fake_imgs = F.interpolate(fake_imgs, size=(target_height, target_width), mode='bilinear', align_corners=False)
            hr_imgs = F.interpolate(hr_imgs, size=(target_height, target_width), mode='bilinear', align_corners=False)
            
            # Get OCR predictions
            fake_preds = self.ocr_model.predict(fake_imgs)
            hr_preds = self.ocr_model.predict(hr_imgs)
            
            # Log shapes for debugging
            logging.debug(f"Fake predictions shape: {fake_preds.shape}")
            logging.debug(f"HR predictions shape: {hr_preds.shape}")
            
            # Ensure predictions have the same shape
            if fake_preds.size() != hr_preds.size():
                min_size = min(fake_preds.size(0), hr_preds.size(0))
                fake_preds = fake_preds[:min_size]
                hr_preds = hr_preds[:min_size]
            
            # Calculate loss
            return self.mse_loss(fake_preds, hr_preds)
        except Exception as e:
            logging.warning(f"OCR loss calculation failed: {str(e)}")
            # Log more details about the error
            logging.debug(f"Input shapes - Fake: {fake_imgs.shape}, HR: {hr_imgs.shape}")
            return torch.tensor(0.0, device=self.device)
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training and validation metrics."""
        logging.info(
            f"Epoch {epoch+1} - "
            f"Train G_Loss: {train_metrics['g_loss']:.4f}, "
            f"D_Loss: {train_metrics['d_loss']:.4f}, "
            f"PSNR: {train_metrics['psnr']:.2f} - "
            f"Val PSNR: {val_metrics['psnr']:.2f}, "
            f"SSIM: {val_metrics['ssim']:.4f}"
        )
    
    def _save_checkpoint(self, epoch: int, psnr: float):
        """Save model checkpoint."""
        # Save best model
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'psnr': psnr
            }, os.path.join(self.checkpoint_dir, 'best_model.pth'))
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'psnr': psnr
            }, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def _save_emergency_checkpoint(self, batch_idx: int):
        """Save emergency checkpoint in case of error."""
        try:
            emergency_path = os.path.join(self.checkpoint_dir, f'emergency_checkpoint_batch_{batch_idx}.pth')
            torch.save({
                'batch_idx': batch_idx,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            }, emergency_path)
            logging.info(f"Emergency checkpoint saved at {emergency_path}")
        except Exception as e:
            logging.error(f"Failed to save emergency checkpoint: {str(e)}") 