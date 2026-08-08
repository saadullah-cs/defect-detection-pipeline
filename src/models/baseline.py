import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

class DefectClassifier(nn.Module):
    """
    Baseline classification model utilizing a pre-trained backbone.
    Adapted dynamically for 1-channel (grayscale) industrial imaging.
    """
    
    def __init__(
        self, 
        num_classes: int = 6, 
        model_name: str = "resnet50", 
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        
        if model_name.lower() == "resnet50":
            self._init_resnet50(num_classes, pretrained)
        else:
            logger.error(f"Architecture '{model_name}' is not supported.")
            raise ValueError(f"Only 'resnet50' is currently implemented. Received: {model_name}")

    def _init_resnet50(self, num_classes: int, pretrained: bool) -> None:
        """Initializes and modifies the ResNet50 architecture."""
        # REMOVE: models.ResNet50_Weights.DEFAULT is the modern PyTorch standard. It pulls the best available ImageNet weights.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # 1. Adapt the first convolutional layer for 1-channel (grayscale) input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # 2. Preserve pre-trained edge detection by averaging the 3-channel weights into the 1-channel layer
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
                
        logger.info("Successfully adapted ResNet50 conv1 layer for 1-channel input.")
        
        # 3. Replace the classification head for our specific defect categories
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # REMOVE: Optional optimization - you can freeze early layers here if compute is tight, but since you have cloud compute, full fine-tuning yields better results for industrial textures.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W).
            
        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)