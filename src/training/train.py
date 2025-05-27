import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from ..models.srrgan import SRRGAN
from ..models.ocr import OCRModule
from ..data.dataset import create_dataloaders, PairedTextImageDataset

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model = SRRGAN().to(self.device)
        self.ocr = OCRModule(device=self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        
        # Initialize schedulers
        self.g_scheduler = optim.lr_scheduler.StepLR(
            self.g_optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        self.d_scheduler = optim.lr_scheduler.StepLR(
            self.d_optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config['log_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
        )
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lr_imgs, hr_imgs, texts) in enumerate(pbar):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Train discriminator
            self.d_optimizer.zero_grad()
            fake_imgs = self.model.generator(lr_imgs)
            d_loss = self.model.get_discriminator_loss(hr_imgs, fake_imgs)
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train generator
            self.g_optimizer.zero_grad()
            fake_imgs = self.model.generator(lr_imgs)
            g_loss = self.model.get_generator_loss(hr_imgs, fake_imgs)
            
            # Add OCR loss
            ocr_loss = self.get_ocr_loss(fake_imgs, texts)
            g_loss = g_loss + self.config['ocr_loss_weight'] * ocr_loss
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Update progress bar
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            pbar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
            
            # Log to tensorboard
            step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/Generator', g_loss.item(), step)
            self.writer.add_scalar('Loss/Discriminator', d_loss.item(), step)
            
            # Save sample images
            if batch_idx % self.config['sample_interval'] == 0:
                self.save_sample_images(lr_imgs, hr_imgs, fake_imgs, epoch, batch_idx)
        
        return total_g_loss / len(train_loader), total_d_loss / len(train_loader)
    
    def get_ocr_loss(self, images, texts):
        """Calculate OCR loss between predicted and ground truth text."""
        # Get OCR predictions
        ocr_outputs = self.ocr.predict(images)
        predicted_texts = self.ocr.decode_predictions(ocr_outputs)
        
        # Calculate loss (simplified version - you'll need to implement proper text comparison)
        loss = torch.tensor(0.0, device=self.device)
        for pred, true in zip(predicted_texts, texts):
            # Implement proper text comparison here
            pass
        
        return loss
    
    def save_sample_images(self, lr_imgs, hr_imgs, fake_imgs, epoch, batch_idx):
        """Save sample images for visualization."""
        # Convert tensors to images and save
        sample_dir = os.path.join(self.config['sample_dir'], f'epoch_{epoch}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save a few sample images
        for i in range(min(4, lr_imgs.size(0))):
            torchvision.utils.save_image(
                lr_imgs[i],
                os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_lr.png')
            )
            torchvision.utils.save_image(
                hr_imgs[i],
                os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_hr.png')
            )
            torchvision.utils.save_image(
                fake_imgs[i],
                os.path.join(sample_dir, f'batch_{batch_idx}_sample_{i}_sr.png')
            )
    
    def save_checkpoint(self, epoch, g_loss, d_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        
        path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, path)
    
    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        for epoch in range(self.config['epochs']):
            # Train one epoch
            g_loss, d_loss = self.train_epoch(train_loader, epoch)
            
            # Update learning rates
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch, g_loss, d_loss)
            
            # Log epoch metrics
            self.writer.add_scalar('Epoch/Generator_Loss', g_loss, epoch)
            self.writer.add_scalar('Epoch/Discriminator_Loss', d_loss, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                for metric_name, value in val_metrics.items():
                    self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
        
        self.writer.close()

def main():
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'lr_step_size': 100,
        'lr_gamma': 0.5,
        'epochs': 100,
        'batch_size': 16,
        'checkpoint_interval': 5,
        'sample_interval': 100,
        'ocr_loss_weight': 0.1,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'sample_dir': 'samples'
    }
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dir='data/train',
        val_dir='data/val',
        batch_size=config['batch_size']
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main() 