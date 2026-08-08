import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Configure module-level logger
logger = logging.getLogger(__name__)

class NEUSurfaceDefectDataset(Dataset):
    """
    PyTorch Dataset for the NEU Surface Defect Database.
    
    Expects a directory structure where subdirectories represent class names:
    root_dir/
        ├── crazing/
        ├── inclusion/
        ├── patches/
        ├── pitted_surface/
        ├── rolled-in_scale/
        └── scratches/
    """
    
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            root_dir (str): Path to the extracted dataset directory.
            transform (Callable, optional): Optional PyTorch transforms to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if not self.root_dir.exists():
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")
            
        # Dynamically infer classes from folder structure to ensure extensibility
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Indexes all image paths and maps them to their respective labels."""
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            # Handle both standard jpg and potentially other formats if the dataset updates
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            
            for file_path in cls_dir.iterdir():
                if file_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(file_path)
                    self.labels.append(self.class_to_idx[cls_name])
                    
        logger.info(f"Loaded {len(self.image_paths)} images across {len(self.classes)} classes.")
        

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads an image from disk, converts to grayscale (as per NEU specs), 
        applies transformations, and returns the tensor-label pair.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Enforce grayscale (L mode) explicitly to prevent 3-channel artifacts
            image = Image.open(img_path).convert("L")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            raise
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Constructs the preprocessing and augmentation pipeline.
    
    Args:
        is_train (bool): If True, applies data augmentation. If False, only resizes and normalizes.
        
    Returns:
        transforms.Compose: The configured transformation pipeline.
    """
    base_transforms = [
        transforms.Resize((224, 224)), # Standard resolution for ResNet/VGG
    ]
    
    if is_train:
        # Augmentations to simulate factory lighting and orientation variances
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) 
        ]
        base_transforms.extend(augmentation_transforms)
        
    tensor_transforms = [
        transforms.ToTensor(),
        # Normalize based on ImageNet stats (adapted for 1-channel grayscale by duplicating the mean/std)
        transforms.Normalize(mean=[0.485], std=[0.229]) 
    ]
    
    base_transforms.extend(tensor_transforms)
    return transforms.Compose(base_transforms)