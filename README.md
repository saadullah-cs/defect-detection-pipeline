# NexGen QA: Industrial Anomaly Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Enterprise-EE4C2C?style=for-the-badge&logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-Hardware_Optimized-005CED?style=for-the-badge&logo=onnx)

An unsupervised machine learning architecture engineered for real-time surface defect detection in manufacturing environments. This pipeline bypasses the need for massive labeled defect datasets by learning the latent spatial representations of "perfect" structural components.

## Architecture Overview

This repository houses the core algorithmic engine for the NexGen QA platform. Rather than using standard supervised classification, this pipeline utilizes a **Convolutional Autoencoder (CAE)**. 

By training exclusively on pristine, defect-free imagery (via the MVTec AD dataset), the network learns a highly compressed latent space. During inference, anomalous regions (scratches, dents, contamination) fail to reconstruct accurately. We extract these pixel-wise discrepancies to generate high-fidelity error maps and precision bounding boxes.

### Core Capabilities
* **Unsupervised Learning:** Zero requirements for labeled anomaly data.
* **Latent Space Compression:** 128-dimensional bottleneck for spatial feature extraction.
* **Hardware-Agnostic Export:** Automated compilation of the PyTorch computation graph to ONNX for zero-latency edge deployment (TensorRT/OpenVINO).

## 📂 Project Topology

```text
├── data/                  # Data ingestion and augmentation pipelines
├── notebooks/             # EDA and experimental model prototyping
├── scripts/               # Core execution layers
│   ├── train_anomaly.py   # Autoencoder training loop & latent mapping
│   ├── infer_anomaly.py   # Y-Offset collision arrays & bounding box extraction
│   └── export_onnx.py     # Graph freezing and ONNX C++ API compilation
└── src/                   # Neural network domain logic
    ├── models/            # ResNet baselines & Autoencoder architectures
    └── utils/             # Evaluation metrics and tensor manipulation
