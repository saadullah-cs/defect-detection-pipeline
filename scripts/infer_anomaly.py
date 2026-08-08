import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# REMOVE: Execute this using: python -m scripts.infer_anomaly
from src.config import Config
from src.models.autoencoder import AnomalyAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Production-ready inference engine for unsupervised anomaly detection.
    Computes reconstruction error maps and extracts bounding boxes.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.DEVICE
        
        # Initialize model
        self.model = AnomalyAutoencoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(self.device)
        self._load_weights()
        self.model.eval()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((cfg.MVTEC_IMAGE_SIZE, cfg.MVTEC_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        
        # Anomaly threshold (can be calibrated dynamically in a real factory using a validation set)
        self.threshold = 0.05 

    def _load_weights(self) -> None:
        checkpoint_path = self.cfg.BASE_DIR / "checkpoints" / f"best_autoencoder_{self.cfg.MVTEC_CATEGORY}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the Autoencoder first.")
            
        logger.info(f"Loading Autoencoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Runs the full anomaly detection pipeline on a single image.
        """
        try:
            original_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            return {"error": str(e)}

        original_size = original_image.size # (width, height)
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed, _ = self.model(input_tensor)
            
            # Calculate pixel-wise Mean Squared Error map
            error_map = F.mse_loss(reconstructed, input_tensor, reduction='none')
            # Average across the color channels (C) to get a 2D spatial heatmap
            error_map = error_map.mean(dim=1).squeeze().cpu().numpy()

        # Generate Bounding Boxes
        bounding_boxes, max_error = self._extract_bounding_boxes(error_map, original_size)
        
        defect_detected = max_error > self.threshold
        
        # Construct JSON Payload matching Section 4 API requirements
        payload = {
            "defect_detected": bool(defect_detected),
            "defect_type": "unknown_anomaly" if defect_detected else "none",
            "confidence_score": float(max_error) if defect_detected else 1.0 - float(max_error),
            "bounding_box_coordinates": bounding_boxes  # List of [x_min, y_min, width, height]
        }
        
        return payload

    def _extract_bounding_boxes(self, error_map: np.ndarray, original_size: tuple) -> tuple:
        """
        Converts the error heatmap into actionable bounding boxes.
        """
        # Normalize error map to 0-255 for OpenCV processing
        norm_error_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Gaussian blur to smooth out noise (dust, minor lighting shifts)
        blurred_map = cv2.GaussianBlur(norm_error_map, (11, 11), 0)
        
        # Threshold the map to create a binary mask of anomalies
        _, binary_mask = cv2.threshold(blurred_map, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        max_error_score = float(np.max(error_map))
        
        scale_x = original_size[0] / self.cfg.MVTEC_IMAGE_SIZE
        scale_y = original_size[1] / self.cfg.MVTEC_IMAGE_SIZE

        for contour in contours:
            # Filter out tiny specks of noise
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale bounding boxes back to original image dimensions
                scaled_bbox = [
                    int(x * scale_x), 
                    int(y * scale_y), 
                    int(w * scale_x), 
                    int(h * scale_y)
                ]
                bboxes.append(scaled_bbox)
                
        return bboxes, max_error_score

def main():
    cfg = Config()
    detector = AnomalyDetector(cfg)
    
    # REMOVE: Point this to a test anomaly image to verify the pipeline
    test_image_path = cfg.MVTEC_DATA_DIR / cfg.MVTEC_CATEGORY / "test" / "broken_large" / "000.png"
    
    if test_image_path.exists():
        logger.info(f"Processing test image: {test_image_path}")
        result_payload = detector.process_image(test_image_path)
        
        print("\n" + "="*50)
        print("API JSON PAYLOAD")
        print("="*50)
        print(json.dumps(result_payload, indent=4))
    else:
        logger.error(f"Test image not found at {test_image_path}. Please adjust the path to a valid image.")

if __name__ == "__main__":
    main()