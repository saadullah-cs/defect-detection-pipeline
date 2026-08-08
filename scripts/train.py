import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# REMOVE: We absolute import from src based on the Python path. 
# Run this script from the project root using: python -m scripts.train
from src.config import Config
from src.data.dataset import NEUSurfaceDefectDataset, get_transforms
from src.models.baseline import DefectClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    cfg = Config()
    
    # Initialize W&B telemetry for cloud monitoring
    wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        config=vars(cfg),
        name=f"{cfg.MODEL_NAME}-baseline"
    )
    
    logger.info(f"Initializing training on device: {cfg.DEVICE}")
    
    # 1. Data Preparation
    # REMOVE: Ensure your raw dataset is extracted at the path defined in cfg.RAW_DATA_DIR
    dataset = NEUSurfaceDefectDataset(
        root_dir=cfg.RAW_DATA_DIR,
        transform=get_transforms(is_train=True)
    )
    
    # 80/20 Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override validation transforms to remove training augmentations
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )
    
    # 2. Model, Loss, and Optimizer setup
    model = DefectClassifier(num_classes=cfg.NUM_CLASSES, model_name=cfg.MODEL_NAME).to(cfg.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # REMOVE: CosineAnnealing safely aggressively reduces LR as we approach the final epochs, stabilizing convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    
    # Mixed precision scaler for optimized VRAM usage on modern cloud GPUs (A100/T4)
    scaler = GradScaler('cuda' if cfg.DEVICE == 'cuda' else 'cpu')
    
    best_val_loss = float('inf')
    save_dir = cfg.BASE_DIR / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    
    # 3. Core Training Loop
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        # tqdm progress bar for local visibility; agents will just log standard output
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Automatic Mixed Precision
            with autocast('cuda' if cfg.DEVICE == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct / train_size
        
        # 4. Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Val]"):
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                
                with autocast('cuda' if cfg.DEVICE == 'cuda' else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct / val_size
        
        logger.info(
            f"Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}"
        )
        
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 5. Model Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = save_dir / f"best_{cfg.MODEL_NAME}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved new best model to {checkpoint_path}")
            
    wandb.finish()

if __name__ == "__main__":
    main()