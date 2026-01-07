# Visualizing Vision Transformers: Attention Rollout vs. Grad-CAM

A comparative study exploring the interpretability differences between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) on texture recognition tasks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![Timm](https://img.shields.io/badge/Library-timm-orange)

## Overview

Understanding *how* deep learning models make decisions is as important as their accuracy, especially in safety-critical applications. This project investigates the distinct "ways of seeing" inherent to two dominant architectures:
1.  **ResNet-50 (CNN):** Relying on local receptive fields and hierarchical feature extraction.
2.  **DeiT-Tiny (ViT):** Relying on global self-attention mechanisms.

Using the **Describable Textures Dataset (DTD)**, this notebook visualizes and contrasts the focus areas of both models using **Grad-CAM** (for CNNs) and **Attention Rollout** (for ViTs).

##  Key Features

* **Custom Attention Rollout:** Manual implementation of the attention flow algorithm (Abnar & Zuidema, 2020) to visualize how information propagates through Transformer layers.
* **State-of-the-Art Tooling:** Leverages `timm` (PyTorch Image Models) for efficient model instantiation and `einops` for tensor manipulation.
* **Head Fusion Analysis:** Visualizes the mean attention across multiple heads to understand the global context the ViT attends to.
* **Comparative XAI:** Side-by-side visualization of ResNet class activation maps vs. ViT attention maps.

## Methodology

### 1. The Models
We utilize pre-trained models fine-tuned on the DTD dataset:
* `resnet50`
* `deit_tiny_patch16_224`

### 2. Visualization Techniques
* **For ResNet:** We use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to highlight the regions in the image that causally influence the classification score.
* **For ViT:** We compute the **Attention Rollout**. This involves recursively multiplying attention matrices across layers to determine the flow of information from input tokens to the final class token.

##  Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sneha-S-D/ViT-vs-CNN.git
    cd ViT-vs-CNN
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision timm einops grad-cam matplotlib
    ```

3.  **Run the Notebook:**
    Open `ViT-vs-CNN.ipynb` in Jupyter or Google Colab to reproduce the experiments.

## Results & Observations

* **CNNs (ResNet):** Tend to focus on specific, discriminative textures (e.g., the distinct pattern of a honeycomb).
* **ViTs (DeiT):** Exhibit a more global understanding, often attending to the shape boundaries and the relationship between texture patches.

<img width="1415" height="345" alt="image" src="https://github.com/user-attachments/assets/47428f6a-2002-442d-b201-4100ba7ea486" />


