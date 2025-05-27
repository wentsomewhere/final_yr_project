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
from src.models.srrgan import Generator, Discriminator, VGGFeatureExtractor
from src.models.ocr import CRNN
from src.utils.metrics import calculate_psnr, calculate_ssim
import torch.nn.functional as F

class Trainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        ocr_model: CRNN,
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
        self.ocr_model = ocr_model.to(device)
        self.feature_extractor = VGGFeatureExtractor().to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
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
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_psnr = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Get data
            lr_imgs, hr_imgs, _ = batch  # Unpack the tuple
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
                'PSNR': f'{psnr:.2f}'
            })
        
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
        try:
            g_loss_ocr = self._calculate_ocr_loss(fake_imgs, hr_imgs)
        except Exception as e:
            logging.warning(f"OCR loss calculation failed: {str(e)}")
            g_loss_ocr = torch.tensor(0.0, device=self.device)
        
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
        # Resize images to CRNN input size (32 height)
        fake_imgs = F.interpolate(fake_imgs, size=(32, 256), mode='bilinear', align_corners=False)
        hr_imgs = F.interpolate(hr_imgs, size=(32, 256), mode='bilinear', align_corners=False)
        
        # Get OCR predictions
        fake_preds = self.ocr_model(fake_imgs)
        hr_preds = self.ocr_model(hr_imgs)
        
        # Ensure predictions have the same shape
        if fake_preds.size() != hr_preds.size():
            min_size = min(fake_preds.size(0), hr_preds.size(0))
            fake_preds = fake_preds[:min_size]
            hr_preds = hr_preds[:min_size]
        
        # Calculate loss
        return self.mse_loss(fake_preds, hr_preds)
    
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