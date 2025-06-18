# 🧠 U-Net for Neuron Segmentation in Biomedical Imaging

This project implements a **U-Net** architecture using PyTorch to perform **semantic segmentation** of neurons in electron microscopy (EM) images. U-Net is a type of convolutional neural network (CNN) that has proven highly effective in biomedical image segmentation tasks.

---

## 🧬 Project Goals

- Segment neurons from high-resolution biomedical images.
- Understand and implement U-Net’s encoder-decoder architecture with skip connections.
- Train and evaluate on real-world electron microscopy datasets.
- Explore image preprocessing, data augmentation, and model training pipelines for segmentation.

---

## 📚 Learning Objectives

By working through this project, you will:

- Learn how U-Net functions and how it is implemented from scratch.
- Understand image segmentation techniques using supervised learning.
- Gain hands-on experience with biomedical image datasets.
- Practice PyTorch module design and model training workflows.

---

## 📊 Dataset

The dataset consists of:

- `train-volume.tif`: A 3D stack of grayscale EM images (slices).
- `train-labels.tif`: Ground truth binary masks of neuron segments.

Each image is:
- Grayscale (single channel)
- Shape: Typically 512x512 pixels (cropped to match output size)

The dataset is assumed to be located in:


---

## 🔧 Project Structure

The architecture includes several modular building blocks:

### 🔹 Contracting Block (Encoder)

Performs downsampling with:
- Two 3×3 convolutions + ReLU
- 2×2 max pooling

Doubles the number of channels at each step.

### 🔹 Expanding Block (Decoder)

Performs upsampling with:
- Bilinear upsampling
- 2×2 convolution to reduce channels
- Skip connection from encoder
- Two 3×3 convolutions + ReLU

### 🔹 Feature Map Blocks

- Initial block maps input to first encoder channel count.
- Final block maps decoder output to desired number of segmentation classes (e.g., 1 for binary segmentation).

---

## 🧱 Model Architecture: U-Net
Input Image ──► FeatureMapBlock ──► ContractingBlock x4 ──► ExpandingBlock x4 ──► FeatureMapBlock ──► Output Mask
↓ skip connect ↓ encoder path ↑ decoder path ↑
downsampled features upsampled features

Each downsampling step halves the spatial size and doubles the number of channels. Each upsampling step reverses that.

---

## 🏁 Getting Started

### ▶️ Requirements

- Python 3.8+
- PyTorch
- NumPy
- matplotlib
- tqdm
- imageio
- scikit-image

### 📦 Install Dependencies

```bash
pip install torch torchvision numpy matplotlib imageio scikit-image tqdm
```
