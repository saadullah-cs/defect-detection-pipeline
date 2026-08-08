import logging
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

class MVTecDataset(Dataset):
    """
    PyTorch Dataset for the MVTec Anomaly Detection Database.
    
    Expected directory structure for a single category (e.g., 'bottle'):
    bottle/
        ├── train/
        │   └── good/              # Only normal images for training
        ├── test/
        │   ├── good/
        │   ├── broken_large/      # Anomaly images
        │   └── contamination/     # Anomaly images
        └── ground_truth/
            ├── broken_large/      # Pixel-level masks
            └── contamination/     # Pixel-level masks
    """

    def __init__(
        self, 
        root_dir: str, 
        category: str, 
        is_train: bool = True, 
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir (str): Base path to the MVTec dataset.
            category (str): Specific object/texture category (e.g., 'cable', 'metal_nut').
            is_train (bool): If True, loads only 'good' training images. If False, loads test set.
            transform (Callable, optional): Transforms to apply to the input image.
            mask_transform (Callable, optional): Transforms to apply to the ground truth mask.
        """
        self.dataset_dir = Path(root_dir) / category
        self.is_train = is_train
        self.transform = transform
        self.mask_transform = mask_transform
        
        if not self.dataset_dir.exists():
            logger.error(f"MVTec category directory not found: {self.dataset_dir}")
            raise FileNotFoundError(f"Directory {self.dataset_dir} does not exist.")
            
        self.image_paths: List[Path] = []
        self.labels: List[int] = []          # 0 for normal, 1 for anomaly
        self.mask_paths: List[Optional[Path]] = [] # None for normal images
        
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Indexes images and maps them to their respective masks and binary labels."""
        phase = 'train' if self.is_train else 'test'
        phase_dir = self.dataset_dir / phase
        
        for defect_type_dir in phase_dir.iterdir():
            if not defect_type_dir.is_dir():
                continue
                
            is_anomaly = defect_type_dir.name != 'good'
            label = 1 if is_anomaly else 0
            
            for img_path in defect_type_dir.rglob('*.png'):
                self.image_paths.append(img_path)
                self.labels.append(label)
                
                # Resolve corresponding ground truth mask for anomalous test images
                if is_anomaly:
                    # MVTec masks have '_mask' appended to the filename
                    mask_filename = f"{img_path.stem}_mask.png"
                    mask_path = self.dataset_dir / 'ground_truth' / defect_type_dir.name / mask_filename
                    
                    if not mask_path.exists():
                        logger.warning(f"Missing ground truth mask: {mask_path}")
                        
                    self.mask_paths.append(mask_path)
                else:
                    self.mask_paths.append(None)
                    
        logger.info(f"Loaded {len(self.image_paths)} images for MVTec category '{self.dataset_dir.name}' Phase: {phase.upper()}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            Tuple containing:
                - image (torch.Tensor): The input image.
                - label (int): 0 for good, 1 for anomaly.
                - mask (torch.Tensor): Ground truth mask (all zeros for 'good' images).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            raise
            
        # Handle Masks: Create an empty (black) mask if the image is 'good'
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, color=0)
            
        if self.transform:
            image = self.transform(image)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)
        elif self.transform:
            # Fallback: Apply identical spatial transforms to the mask if specific mask transforms aren't provided
            # Note: We must ensure no color jittering or normalization is applied to the binary mask.
            # In a production pipeline, it is safer to define a distinct mask_transform.
            mask = self.transform(mask)
            
        return image, label, mask