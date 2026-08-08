import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class NEUSurfaceDefectDataset(Dataset):
    """
    PyTorch Dataset adapted for the NEU-DET directory structure.
    Expects: data/raw/{phase}/images/
    """
    def __init__(self, root_dir: str, phase: str = 'train', transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir) / phase / "images"
        self.transform = transform
        
        # Hardcode classes since we are inferring them from filenames
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        if not self.root_dir.exists():
            logger.error(f"Dataset directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")
            
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        for file_path in self.root_dir.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                # In NEU-DET, filenames start with their class (e.g., 'crazing_1.jpg')
                for cls_name in self.classes:
                    if file_path.name.startswith(cls_name):
                        self.image_paths.append(file_path)
                        self.labels.append(self.class_to_idx[cls_name])
                        break
                        
        logger.info(f"Loaded {len(self.image_paths)} images from {self.root_dir.parent.name} split.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
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
    """
    base_transforms = [
        transforms.Resize((224, 224)),
    ]
    
    if is_train:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) 
        ]
        base_transforms.extend(augmentation_transforms)
        
    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) 
    ]
    
    base_transforms.extend(tensor_transforms)
    return transforms.Compose(base_transforms)