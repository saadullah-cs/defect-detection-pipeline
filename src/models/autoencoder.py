import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Compresses the input image into a latent, lower-dimensional representation.
    Extracts structural features of 'normal' defect-free surfaces.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # REMOVE: MVTec AD contains both RGB and Grayscale categories. We default to 3 channels (RGB) for flexibility.
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x)


class Decoder(nn.Module):
    """
    Reconstructs the image from the latent representation.
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # REMOVE: We use ConvTranspose2d (Deconvolution) to upsample back to the original image dimensions.
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Bound pixel values between [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv_blocks(x)


class AnomalyAutoencoder(nn.Module):
    """
    End-to-end Autoencoder for unsupervised anomaly detection and localization.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        
        logger.info(f"Initialized Anomaly Autoencoder with latent dimension: {latent_dim}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Reconstructed image (B, C, H, W).
                - Latent representation (useful for downstream tasks if needed).
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent