import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import Levenshtein
from typing import List, Dict, Union

class MetricsCalculator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_image_metrics(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> Dict[str, float]:
        """Calculate image quality metrics between super-resolved and high-resolution images."""
        # Convert tensors to numpy arrays
        sr_img = sr_img.cpu().numpy()
        hr_img = hr_img.cpu().numpy()
        
        # Calculate PSNR
        psnr_value = psnr(hr_img, sr_img)
        
        # Calculate SSIM
        ssim_value = ssim(hr_img, sr_img, multichannel=True)
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        }
    
    def calculate_ocr_metrics(self, predicted_texts: List[str], ground_truth_texts: List[str]) -> Dict[str, float]:
        """Calculate OCR performance metrics."""
        total_chars = 0
        total_words = 0
        char_errors = 0
        word_errors = 0
        
        for pred, true in zip(predicted_texts, ground_truth_texts):
            # Character Error Rate (CER)
            char_errors += Levenshtein.distance(pred, true)
            total_chars += len(true)
            
            # Word Error Rate (WER)
            pred_words = pred.split()
            true_words = true.split()
            word_errors += Levenshtein.distance(pred_words, true_words)
            total_words += len(true_words)
        
        cer = char_errors / total_chars if total_chars > 0 else 1.0
        wer = word_errors / total_words if total_words > 0 else 1.0
        
        return {
            'cer': cer,
            'wer': wer,
            'accuracy': 1 - wer
        }
    
    def calculate_all_metrics(self, 
                            sr_imgs: torch.Tensor, 
                            hr_imgs: torch.Tensor,
                            predicted_texts: List[str],
                            ground_truth_texts: List[str]) -> Dict[str, float]:
        """Calculate all metrics for the model evaluation."""
        # Calculate image metrics
        image_metrics = self.calculate_image_metrics(sr_imgs, hr_imgs)
        
        # Calculate OCR metrics
        ocr_metrics = self.calculate_ocr_metrics(predicted_texts, ground_truth_texts)
        
        # Combine all metrics
        all_metrics = {**image_metrics, **ocr_metrics}
        
        # Update stored metrics
        self.metrics = all_metrics
        
        return all_metrics
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get a summary of all calculated metrics."""
        return self.metrics

class BenchmarkingReport:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_calculator = MetricsCalculator()
        self.comparison_results = {}
    
    def add_comparison(self, 
                      method_name: str,
                      sr_imgs: torch.Tensor,
                      hr_imgs: torch.Tensor,
                      predicted_texts: List[str],
                      ground_truth_texts: List[str]):
        """Add comparison results for a specific method."""
        metrics = self.metrics_calculator.calculate_all_metrics(
            sr_imgs, hr_imgs, predicted_texts, ground_truth_texts
        )
        self.comparison_results[method_name] = metrics
    
    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """Generate a comprehensive benchmarking report."""
        return {
            'model_name': self.model_name,
            'comparisons': self.comparison_results
        }
    
    def save_report(self, filepath: str):
        """Save the benchmarking report to a file."""
        import json
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)

def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: str = 'cuda') -> Dict[str, float]:
    """Evaluate model performance on a dataset."""
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    all_sr_imgs = []
    all_hr_imgs = []
    all_predicted_texts = []
    all_ground_truth_texts = []
    
    with torch.no_grad():
        for lr_imgs, hr_imgs, texts in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate super-resolved images
            sr_imgs = model(lr_imgs)
            
            # Get OCR predictions
            predicted_texts = model.ocr.decode_predictions(model.ocr.predict(sr_imgs))
            
            # Store results
            all_sr_imgs.append(sr_imgs)
            all_hr_imgs.append(hr_imgs)
            all_predicted_texts.extend(predicted_texts)
            all_ground_truth_texts.extend(texts)
    
    # Concatenate all images
    all_sr_imgs = torch.cat(all_sr_imgs, dim=0)
    all_hr_imgs = torch.cat(all_hr_imgs, dim=0)
    
    # Calculate metrics
    metrics = metrics_calculator.calculate_all_metrics(
        all_sr_imgs, all_hr_imgs,
        all_predicted_texts, all_ground_truth_texts
    )
    
    return metrics 