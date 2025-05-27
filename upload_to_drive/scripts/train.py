import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.srrgan import Generator, Discriminator
from src.models.ocr import OCRModule
from src.data.dataset import PairedTextImageDataset
from src.utils.trainer import Trainer

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Train SRR-GAN model')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the datasets')
    parser.add_argument('--models_dir', type=str, default='models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0,
                      help='Weight for perceptual loss')
    parser.add_argument('--lambda_ocr', type=float, default=1.0,
                      help='Weight for OCR loss')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Loading datasets...")
    
    # Create datasets
    train_dataset = PairedTextImageDataset(
        [os.path.join(args.data_dir, 'mnist', 'train'), os.path.join(args.data_dir, 'printed', 'train')],
        scale_factor=4
    )
    val_dataset = PairedTextImageDataset(
        [os.path.join(args.data_dir, 'mnist', 'val'), os.path.join(args.data_dir, 'printed', 'val')],
        scale_factor=4
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    logging.info("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator()
    discriminator = Discriminator()
    ocr_model = OCRModule(device=device)
    
    # Initialize trainer
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        ocr_model=ocr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        lambda_perceptual=args.lambda_perceptual,
        lambda_ocr=args.lambda_ocr,
        checkpoint_dir=args.models_dir
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.train(num_epochs=args.num_epochs)

if __name__ == '__main__':
    main() 