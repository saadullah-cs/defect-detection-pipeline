import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.config import Config
from src.data.dataset import NEUSurfaceDefectDataset, get_transforms
from src.models.baseline import DefectClassifier
from src.utils.metrics import save_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    cfg = Config()
    
    # 1. Load Validation Dataset natively
    val_dataset = NEUSurfaceDefectDataset(
        root_dir=cfg.RAW_DATA_DIR,
        phase='validation',
        transform=get_transforms(is_train=False)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.DEVICE == "cuda" else False
    )
    
    # 2. Load Model Checkpoint
    checkpoint_path = cfg.BASE_DIR / "checkpoints" / f"best_{cfg.MODEL_NAME}.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        return

    model = DefectClassifier(num_classes=cfg.NUM_CLASSES, model_name=cfg.MODEL_NAME).to(cfg.DEVICE)
    
    logger.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Inference Engine
    all_preds = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(cfg.DEVICE)
            
            with torch.amp.autocast('cuda' if cfg.DEVICE == 'cuda' else 'cpu'):
                outputs = model(images)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 4. Metric Computation & Reporting
    class_names = val_dataset.classes
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # 5. Generate Visualizations
    reports_dir = cfg.BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    cm_path = reports_dir / "confusion_matrix.png"
    save_confusion_matrix(all_labels, all_preds, class_names, cm_path)

if __name__ == "__main__":
    main()