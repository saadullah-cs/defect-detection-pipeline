> **💡 ARCHITECTURE EVOLUTION:** 
> This repository represents **Phase 1** of the NexGen QA project, utilizing an **Unsupervised Convolutional Autoencoder** trained solely on flawless components to detect anomalies via MSE thresholding.
> 
> For **Phase 2**, we evolved the architecture into a high-speed, **Supervised YOLOv8 ONNX** engine capable of live-streaming WebRTC video feeds. 
> 
> 👉 **[Click here to view the V2 Production Monorepo](https://github.com/saadullah-cs/nexgen-vision-qa)**

# NexGen QA | Industrial Defect Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Enterprise-EE4C2C?style=for-the-badge&logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-Hardware_Optimized-005CED?style=for-the-badge&logo=onnx)

An unsupervised machine learning architecture built for real-time surface defect detection in manufacturing. This pipeline completely bypasses the need for massive labeled defect datasets by learning the latent spatial representations of perfect structural components.

## 🔬 Architecture Overview

This repo contains the core algorithmic training engine. Instead of standard supervised classification, we engineered a Convolutional Autoencoder (CAE) designed for industrial anomaly detection.

By training exclusively on pristine, defect-free imagery, the network learns a highly compressed latent space. During inference, any anomalous regions (scratches, dents, contamination) fail to reconstruct accurately. We extract these pixel-wise discrepancies to generate high-fidelity error maps and precision bounding boxes.

### Core Capabilities
* **Unsupervised Learning:** Zero requirements for labeled anomaly data.
* **Latent Space Compression:** 128-dimensional bottleneck for robust spatial feature extraction.
* **Hardware-Agnostic Export:** Automated compilation of the PyTorch computation graph to ONNX for zero-latency edge deployment via TensorRT or OpenVINO.

## 📐 Project Topology

```text
├── data/                  # Data ingestion and augmentation pipelines
├── notebooks/             # EDA and experimental model prototyping
├── scripts/               # Core execution layers
│   ├── train_anomaly.py   # Autoencoder training loop & latent mapping
│   ├── infer_anomaly.py   # Y-Offset collision arrays & bounding box extraction
│   └── export_onnx.py     # Graph freezing and ONNX API compilation
└── src/                   # Neural network domain logic
    ├── models/            # ResNet baselines & Autoencoder architectures
    └── utils/             # Evaluation metrics and tensor manipulation
```

## ⚡ Execution

**Phase 1: Latent Representation Training**
```bash
python -m scripts.train_anomaly
```

**Phase 2: Hardware Optimization**
```bash
python -m scripts.export_onnx
```
Outputs `.onnx` and `.data` weight files ready for microservice deployment.
