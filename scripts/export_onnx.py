import os
import logging
from pathlib import Path

import torch
import torch.onnx

# REMOVE: Execute this using: python -m scripts.export_onnx
from src.config import Config
from src.models.autoencoder import AnomalyAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_to_onnx(cfg: Config):
    """
    Exports the trained PyTorch Autoencoder to an optimized ONNX format 
    suitable for low-latency edge deployment (e.g., Jetson Nano, Raspberry Pi).
    """
    device = torch.device('cpu') # Exporting is safely done on CPU
    
    # 1. Initialize model and load weights
    model = AnomalyAutoencoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(device)
    checkpoint_path = cfg.BASE_DIR / "checkpoints" / f"best_autoencoder_{cfg.MVTEC_CATEGORY}.pth"
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}. Cannot export.")
        return
        
    logger.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode to disable dropout and freeze batch normalization layers
    model.eval()
    
    # 2. Prepare Dummy Input
    # ONNX requires a sample input to trace the computation graph.
    # Shape: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, cfg.MVTEC_IMAGE_SIZE, cfg.MVTEC_IMAGE_SIZE, device=device)
    
    # 3. Define Export Paths
    export_dir = cfg.BASE_DIR / "deploy"
    export_dir.mkdir(exist_ok=True)
    onnx_file_path = export_dir / f"anomaly_autoencoder_{cfg.MVTEC_CATEGORY}.onnx"
    
    # 4. Export the Model
    logger.info("Tracing computational graph and exporting to ONNX...")
    
    torch.onnx.export(
        model,                             # Model being run
        dummy_input,                       # Model input (or a tuple for multiple inputs)
        onnx_file_path,                    # Where to save the model
        export_params=True,                # Store the trained parameter weights inside the model file
        opset_version=14,                  # Standard opset version for modern deployment
        do_constant_folding=True,          # Optimize constant operations for faster inference
        input_names=['input_image'],       # Define the model's input dictionary keys
        output_names=['reconstructed', 'latent'], # Define the model's output dictionary keys
        dynamic_axes={                     # Allow dynamic batch sizes for flexible edge inference
            'input_image': {0: 'batch_size'},
            'reconstructed': {0: 'batch_size'},
            'latent': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Successfully exported optimized ONNX model to: {onnx_file_path}")
    logger.info("This model can now be loaded via ONNXRuntime, TensorRT, or OpenVINO on edge hardware.")

def main():
    cfg = Config()
    export_to_onnx(cfg)

if __name__ == "__main__":
    main()