# 🧠 U-Net for Neuron Segmentation in Biomedical Imaging

This project implements a U-Net architecture using PyTorch to perform semantic segmentation of neurons in electron microscopy images. The primary objective is to accurately identify and classify pixels that correspond to neural structures within high-resolution biomedical images. Segmenting neurons from electron microscopy data is a fundamental task in biomedical image analysis, especially in fields such as connectomics, where researchers aim to reconstruct and understand the structure and function of complex neural circuits. Traditional image processing methods often struggle with the dense textures, subtle gradients, and irregular boundaries present in these types of images. Deep learning, and particularly convolutional neural networks, have demonstrated superior performance by learning robust and hierarchical feature representations directly from raw image data.

U-Net, the architecture used in this project, was originally proposed in 2015 by Olaf Ronneberger and colleagues. It has become one of the most popular models for biomedical image segmentation due to its unique structure that combines both an encoder and a decoder path. The encoder captures context and abstract features by progressively downsampling the input, while the decoder reconstructs the spatial dimensions and fine details through upsampling. Crucially, U-Net employs skip connections that link corresponding layers in the encoder and decoder paths, allowing fine-grained features to flow directly to the reconstruction layers. This design helps the network preserve spatial information and improves its ability to localize objects with precision. In this project, the U-Net is trained on a set of grayscale volumetric EM images, with corresponding ground truth labels that indicate neuron regions at the pixel level. The model learns to generate probability masks that highlight neural structures, which can then be thresholded or further processed for downstream analysis. This project not only demonstrates how to implement and train U-Net from scratch using PyTorch, but also serves as a valuable foundation for more advanced research in medical image analysis, generative modeling, and computer vision in scientific domains.

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

## 🧰 Tools and Libraries

| Library/Tool         | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **PyTorch**           | Deep learning framework used to build and train the U-Net model            |
| **torchvision**       | Utilities for image transformation and visualization                       |
| **tqdm**              | Provides progress bars for training loops                                  |
| **matplotlib**        | Used for visualizing images and model predictions                          |
| **imageio (v3)**      | Reads TIFF format EM images used as input data                             |
| **scikit-image (skimage)** | Image processing library used for reading and cropping images           |
| **NumPy**             | Supports numerical operations and tensor manipulation                      |
| **Google Colab / CUDA** | Hardware acceleration (optional) for training models with GPU support     |

---

## 📦 Install Dependencies

```bash
pip install torch torchvision numpy matplotlib imageio scikit-image tqdm
```
