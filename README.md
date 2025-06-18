# 🧠 U-Net for Neuron Segmentation in Biomedical Imaging

This project implements a U-Net architecture using PyTorch to perform semantic segmentation of neurons in electron microscopy (EM) images. The primary objective is to accurately identify and classify pixels that correspond to neural structures within high-resolution biomedical images.

Segmenting neurons from EM data is a fundamental task in biomedical image analysis, particularly in connectomics,a field focused on reconstructing and understanding the structure and function of complex neural circuits. Traditional image processing methods often struggle with the dense textures, subtle gradients, and irregular boundaries that characterize EM images. In contrast, deep learning, and specifically convolutional neural networks (CNNs), have demonstrated superior performance by learning hierarchical and robust feature representations directly from raw image data.

U-Net, the core architecture used in this project, was introduced by Olaf Ronneberger et al. in 2015 and has become one of the most widely used models in biomedical segmentation. Its architecture features a symmetric encoder-decoder structure: the encoder captures contextual information by progressively downsampling the image, while the decoder restores spatial resolution through upsampling. Crucially, U-Net includes skip connections between corresponding layers in the encoder and decoder paths, enabling the network to recover fine-grained spatial details lost during downsampling.

In this project, the U-Net is trained on a set of grayscale volumetric EM images along with corresponding binary masks that represent ground truth neuron regions at the pixel level. The network learns to produce probability masks that highlight neural structures within the input images. These masks can be post-processed or thresholded to generate precise segmentations suitable for further biological analysis.

Beyond demonstrating a working implementation of U-Net in PyTorch, this project serves as a valuable foundation for more advanced research in medical image analysis, computer vision, and generative modeling within scientific domains.

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

Each downsampling step halves the spatial size and doubles the number of channels. Each upsampling step reverses that.

---

## 🧪 Results: Progressive Improvement in Neuron Segmentation Using U-Net

The following figure illustrates the improvement in segmentation quality over the course of training the U-Net model for neuron segmentation on electron microscopy (EM) images.

![Screenshot 2025-06-18 172514](https://github.com/user-attachments/assets/5cce1f14-2119-40a6-bfe0-72a6acc7f491)

### 🔍 Image Description

This figure compares segmentation results at two distinct points in the training process:

- **Left Panel: Early Stage (Epoch 10, Step 80)**
  - **Loss**: `0.3461`
  - **Top Row**: Raw EM input images.
  - **Middle Row**: Predicted segmentation masks by the U-Net model.
  - **Bottom Row**: Overlay of EM images and predicted contours.
  - At this early stage, the model struggles to identify neuron boundaries clearly, showing noisy predictions and incomplete structures.

- **Right Panel: Late Stage (Epoch 197, Step 1580)**
  - **Loss**: `0.0480`
  - **Top Row**: EM input images.
  - **Middle Row**: Final predicted segmentation masks.
  - **Bottom Row**: Overlay of EM images and refined contours.
  - The predictions are significantly cleaner and more accurate, with well-defined boundaries and fewer artifacts, reflecting the model’s improved understanding of the neuron morphology.

This result demonstrates that U-Net effectively learns to delineate neural structures as training progresses, confirming its suitability for dense biomedical image segmentation tasks.

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
