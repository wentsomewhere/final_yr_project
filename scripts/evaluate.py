import os
import argparse
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from tqdm import tqdm

from src.models.srrgan import SRRGAN
from src.models.ocr import OCRModule
from src.data.dataset import TextImageDataset
from src.evaluation.metrics import MetricsCalculator

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = SRRGAN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model: SRRGAN, ocr: OCRModule, dataloader: DataLoader,
                  metrics_calculator: MetricsCalculator, device: torch.device):
    """Evaluate model on test dataset."""
    model.eval()
    ocr.eval()
    
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'ocr_accuracy': [],
        'cer': [],
        'wer': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            lr_images = batch['lr_image'].to(device)
            hr_images = batch['hr_image'].to(device)
            ground_truth_text = batch['text']
            
            # Generate enhanced images
            enhanced_images = model(lr_images)
            
            # Calculate image metrics
            batch_metrics = metrics_calculator.calculate_image_metrics(
                enhanced_images, hr_images
            )
            
            # Perform OCR on enhanced images
            ocr_text = ocr(enhanced_images)
            
            # Calculate OCR metrics
            ocr_metrics = metrics_calculator.calculate_ocr_metrics(
                ocr_text, ground_truth_text
            )
            
            # Update metrics
            for metric in all_metrics:
                if metric in batch_metrics:
                    all_metrics[metric].extend(batch_metrics[metric])
                elif metric in ocr_metrics:
                    all_metrics[metric].extend(ocr_metrics[metric])
    
    # Calculate average metrics
    avg_metrics = {
        metric: sum(values) / len(values)
        for metric, values in all_metrics.items()
    }
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate SRR-GAN model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to evaluate on (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.results_dir)
    logging.info("Starting evaluation process...")
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device(args.device)
    
    # Load models
    srrgan = load_model(args.model_path, device)
    ocr = OCRModule().to(device)
    
    # Initialize dataset and dataloader
    dataset = TextImageDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator()
    
    # Evaluate model
    metrics = evaluate_model(srrgan, ocr, dataloader, metrics_calculator, device)
    
    # Log results
    logging.info("Evaluation results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save results
    results_path = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Results saved to {results_path}")

if __name__ == '__main__':
    main() 