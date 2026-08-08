import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# REMOVE: Execute this using: python -m scripts.train_anomaly
from src.config import Config
from src.data.mvtec import MVTecDataset
from src.models.autoencoder import AnomalyAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mvtec_transforms(image_size: int) -> transforms.Compose:
    """Standard transforms for MVTec images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Normalize to [0, 1] range to match the Decoder's Sigmoid output
    ])

def main():
    cfg = Config()
    
    # Initialize W&B for Phase 2
    wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        config=vars(cfg),
        name=f"autoencoder-{cfg.MVTEC_CATEGORY}"
    )
    
    logger.info(f"Initializing Autoencoder training for category: {cfg.MVTEC_CATEGORY}")
    
    # 1. Data Preparation (Train ONLY on good images)
    train_dataset = MVTecDataset(
        root_dir=str(cfg.MVTEC_DATA_DIR),
        category=cfg.MVTEC_CATEGORY,
        is_train=True,
        transform=get_mvtec_transforms(cfg.MVTEC_IMAGE_SIZE)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )
    
    # 2. Model, Loss, and Optimizer
    model = AnomalyAutoencoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(cfg.DEVICE)
    
    # L2 Loss (MSE) ensures the model heavily penalizes large pixel-wise differences
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    scaler = GradScaler('cuda' if cfg.DEVICE == 'cuda' else 'cpu')
    
    best_loss = float('inf')
    save_dir = cfg.BASE_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    
    # 3. Training Loop
    for epoch in range(1, cfg.ANOMALY_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.ANOMALY_EPOCHS} [Train]")
        
        # REMOVE: We ignore the label and mask since training data is 100% normal (defect-free)
        for images, _, _ in pbar:
            images = images.to(cfg.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda' if cfg.DEVICE == 'cuda' else 'cpu'):
                reconstructed, _ = model(images)
                loss = criterion(reconstructed, images)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(train_dataset)
        scheduler.step(avg_loss)
        
        logger.info(f"Epoch {epoch} | Avg Reconstruction Loss: {avg_loss:.6f}")
        wandb.log({
            "epoch": epoch,
            "reconstruction_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 4. Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_dir / f"best_autoencoder_{cfg.MVTEC_CATEGORY}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            logger.info(f"Saved optimized autoencoder to {checkpoint_path}")
            
    wandb.finish()

if __name__ == "__main__":
    main()