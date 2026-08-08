\# Automated Industrial Defect Detection System



!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)

!\[ONNX](https://img.shields.io/badge/ONNX-Optimized-005CED)

!\[License](https://img.shields.io/badge/License-Proprietary-red)



An end-to-end, Industry 4.0 compliant computer vision pipeline designed for identifying, classifying, and localizing manufacturing defects in real-time. 



This project bridges the gap between supervised baseline classification and advanced unsupervised anomaly detection for novel defects, ultimately optimized for low-latency edge inference.



\## System Architecture



The pipeline operates in two distinct phases:

1\. \*\*Phase 1 (Supervised Baseline):\*\* Leverages a ResNet-50 backbone dynamically adapted for 1-channel grayscale imaging to classify 6 known defect categories (NEU dataset).

2\. \*\*Phase 2 (Unsupervised Anomaly Detection):\*\* Employs a Convolutional Autoencoder (CAE) trained exclusively on defect-free products (MVTec AD dataset). Anomalies are localized during inference via a thresholded reconstruction error map.



\## Setup \& Installation



1\. \*\*Clone the repository:\*\*

&#x20;  ```bash

&#x20;  git clone <repository-url>

&#x20;  cd defect-detection-pipeline

