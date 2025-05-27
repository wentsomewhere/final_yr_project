import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        
    Returns:
        PSNR value in dB
    """
    # Ensure images are in range [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Calculate MSE
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        
    Returns:
        SSIM value in range [0, 1]
    """
    # Ensure images are in range [0, 1]
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Create Gaussian window
    window = _create_gaussian_window(window_size, sigma).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def _create_gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian window.
    
    Args:
        window_size: Size of the window
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Gaussian window tensor
    """
    x = torch.arange(window_size, dtype=torch.float32)
    x = x - window_size // 2
    
    # Create 2D coordinates
    x, y = torch.meshgrid(x, x, indexing='ij')
    
    # Calculate Gaussian
    g = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
    
    # Normalize
    g = g / g.sum()
    
    return g

def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[float, float]:
    """Calculate both PSNR and SSIM metrics.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        
    Returns:
        Tuple of (PSNR, SSIM) values
    """
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    return psnr, ssim 