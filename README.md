# 🌿 Plant Disease Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</div>

<p align="center">
  A deep learning project for accurate classification of plant diseases using MobileNetV3 architecture
</p>

## 📋 Overview

This project implements a complete machine learning pipeline for plant disease classification, built using PyTorch and the MobileNetV3 architecture. The system can identify 15 different plant disease classes from the PlantVillage dataset with high accuracy.

### 🔍 Key Features

- **High Accuracy**: Achieves >97% accuracy on test data
- **Fast Inference**: Optimized for quick predictions (milliseconds per image)
- **Interactive UI**: Web interface with GradCAM visualization of model attention
- **Modular Design**: Clean, modular code structure for easy customization
- **Comprehensive Evaluation**: Detailed metrics and visualizations

![System Architecture](https://mermaid.ink/img/pako:eNp1kc9OwzAMxl8l8nkH2DjRJoSgQmOIE5rlkJq0haVN5ThDQ913x0k42DbmU2z_Pn-O5ZELrQkzUeiQu0wWxVOjGNkNWj0B-DcKhNIZFP0WIRVg-Gd2C0YMKJIAHk-hLpYRfNDBfaA0Z6RLVG3rydXI6YIIGRnTc7wRvtjVlOu7oVzPOyEYjCPIW6g0KcSF0-vl_5Lp3ZlDVOaS2nwkMY3aVcYBzs6Xk5Grc0IMNUR2B_USuuR1rZNjbvF8Wkw5ULpwzIg3CW9dPeM0Oa9OuVfWxXq43GFzlaPJoZFt-ZnzLtvXadVT-bWOzvVrC5e0Pu75z1ZCPt9-wEzEqWnPeUeVZppF6Uh35NZUu4iHPGgmO4hScXPYR-IbjLNK6Qgl04rxmDXGdpJDxuAVZfzAG2PY9htGEcog?type=png)

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (recommended)
- CUDA-compatible GPU (optional, but recommended for training)

### 📦 Installation

<details>
<summary><b>Windows Setup</b></summary>

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Linux Setup</b></summary>

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
</details>

## 📊 Project Structure

```
.
├── prepare_dataset.py     # Dataset preparation script
├── train.py               # Model training script 
├── inference.py           # Single image prediction script
├── evaluate.py            # Model evaluation on test dataset
├── app.py                 # Streamlit web interface with GradCAM
├── PlantVillage/          # Original dataset directory
├── PlantVillage_Split/    # Processed dataset with splits
│   ├── train/             # Training data (70%)
│   ├── val/               # Validation data (20%)
│   └── test/              # Test data (10%)
└── outputs/               # Generated outputs
    ├── best_model.pth     # Best model weights
    ├── training_curves.png# Loss and accuracy plots
    └── class_names.txt    # List of class names
```

## 💻 Usage Guide

### 1️⃣ Dataset Preparation

Split your dataset into train, validation, and test sets:

```bash
python prepare_dataset.py --source ./PlantVillage --output ./PlantVillage_Split --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

<details>
<summary>Command-line options</summary>

- `--source`: Source directory containing class folders
- `--output`: Output directory for the split dataset
- `--train-ratio`: Proportion of data for training (default: 0.7)
- `--val-ratio`: Proportion of data for validation (default: 0.2)
- `--test-ratio`: Proportion of data for testing (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
</details>

### 2️⃣ Model Training

Train the classifier with:

```bash
python train.py --model mobilenet_v3_large --batch-size 32 --epochs 20 --lr 0.001
```

<details>
<summary>Command-line options</summary>

- `--data-dir`: Path to dataset directory (default: ./PlantVillage_Split)
- `--img-size`: Input image size (default: 224)
- `--batch-size`: Batch size for training (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--model`: Model architecture (choices: mobilenet_v3_small, mobilenet_v3_large)
- `--pretrained`: Use pretrained weights (default: True)
- `--freeze-backbone`: Train only the classifier head (default: False)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay factor (default: 1e-4)
- `--mixed-precision`: Enable mixed precision training (default: True)
- `--seed`: Random seed (default: 42)
- `--output-dir`: Output directory (default: ./outputs)
</details>

### 3️⃣ Single Image Inference

Test your model on individual images:

```bash
python inference.py --image path/to/your/image.jpg
```

<details>
<summary>Command-line options</summary>

- `--image`: Path to the input image (required)
- `--model-path`: Path to the trained model file (default: ./outputs/best_model.pth)
- `--model-type`: Model architecture (default: mobilenet_v3_large)
- `--class-names`: Path to class names file (default: ./outputs/class_names.txt)
- `--img-size`: Input image size (default: 224)
</details>

### 4️⃣ Model Evaluation

Perform comprehensive evaluation on the test dataset:

```bash
python evaluate.py
```

<details>
<summary>Command-line options</summary>

- `--data-dir`: Path to test data directory (default: ./PlantVillage_Split/test)
- `--model-path`: Path to the model file (default: ./outputs/best_model.pth)
- `--model-type`: Model architecture (default: mobilenet_v3_large)
- `--class-names`: Path to class names file (default: ./outputs/class_names.txt)
- `--img-size`: Input image size (default: 224)
- `--batch-size`: Batch size for evaluation (default: 64)
- `--num-workers`: Number of data loading workers (default: 4)
- `--output-dir`: Directory for evaluation results (default: ./evaluation)
</details>

### 5️⃣ Web Interface

Launch the interactive Streamlit interface with GradCAM visualization:

```bash
# Activate your virtual environment first
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

streamlit run app.py
```

The web interface provides:
- Image upload capability
- Disease classification with confidence scores
- GradCAM visualization of model attention regions
- Inference time measurement
- Top 5 prediction display

![Streamlit Interface Example](https://mermaid.ink/img/pako:eNplkU1rwzAMhv-K0XUZJGtv9aGwHsZg7GOM0YtiK4nBH8F2xyj97_OSlrXdTrL1PpIl6UQ10ZjOZPHDjkdkM7_Bt7pqkOZAJKRhQZzfRtMcxgGNl0x-A8Ia1YbO_rQiJV6p2iHTKKRnKgdMXK5Vhkh3KvGZPdq2rNHEoMWGgLcJOWvhCi1qz2Ib7VAkhtFExvq52u9pLxDYv0jMg2_rMtTREm2-QDZJJAqxEGRqQWQecA1xAJJgIW8yScQ8cT4sQPQtgAueBtMgKZhQvXYyiBbDKNiDezFw05PbzPM8NeXvdtYPcFQ8qULNJNyJz7F_-kf1qcSjN27J2aZ1s1NlbabFLJFdMcNZFTOPdZwK8TgxYTMp-8Z-2a1p8L3_0a8ZVd3UdXtuO6psZNlRm8lCuoQqFHukxmbjVxsm5fpNkv8Go1qpiVDyQcleQE2WRZ3JQ4Q9KPQj7cw43Qbj43c-?type=png)

## 📊 Training Details

The model training pipeline includes:

- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **Mixed Precision**: Faster training with FP16/FP32 precision
- **Learning Rate Scheduler**: Cosine annealing for optimal convergence
- **Checkpointing**: Saves best model based on validation accuracy
- **Visual Monitoring**: Logs and plots for loss and accuracy tracking

## 📈 Evaluation Metrics

The evaluation script produces:

- **Confusion Matrix**: Visual representation of classification performance
- **Excel Report**: Detailed per-class precision, recall, F1-score metrics
- **Terminal Output**: Summary statistics and classification report

## 🎛️ Customization

The project is designed for easy customization:

- Choose between MobileNetV3 variants (small/large)
- Adjust learning rates and optimization parameters
- Modify data augmentation strategies
- Fine-tune a pre-trained model by freezing the backbone
- Change image preprocessing parameters

## 🤝 Contributing

Contributions to improve the project are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- MobileNetV3 paper authors for the model architecture
- PlantVillage dataset for providing labeled plant disease images 