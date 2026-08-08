import os
from dataclasses import dataclass
from pathlib import Path

# REMOVE: Using dataclasses ensures strict typing and prevents accidental modification of config variables during runtime.

@dataclass
class Config:
    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # Model Hyperparameters
    MODEL_NAME: str = "resnet50"
    NUM_CLASSES: int = 6
    IMAGE_SIZE: int = 224
    
    # Training Hyperparameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    
    # Compute
    # Automatically bind to GPU if available in your GitHub Cloud space
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or __import__("torch").cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4
    
    # MLOps & Tracking
    WANDB_PROJECT: str = "neu-surface-defect-detection"
    WANDB_ENTITY: str = None  # Add your team/username here if applicable

    # MVTec Phase 2 Configuration
    MVTEC_DATA_DIR: Path = DATA_DIR / "mvtec"
    MVTEC_CATEGORY: str = "bottle"  # e.g., 'bottle', 'cable', 'hazelnut', 'metal_nut'
    MVTEC_IMAGE_SIZE: int = 256     # Autoencoders usually need dimensions in powers of 2
    LATENT_DIM: int = 128
    ANOMALY_EPOCHS: int = 100       # Autoencoders typically need more epochs to converge