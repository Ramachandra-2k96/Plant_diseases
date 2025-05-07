# Plant Disease Classification

A deep learning project for classifying plant diseases using MobileNetV3 architecture. This project provides a complete pipeline for dataset preparation, model training, inference, evaluation, and visualization.

## Overview

This project uses PyTorch to train a MobileNetV3 model for plant disease classification. The dataset used is the PlantVillage dataset, which contains images of healthy and diseased plant leaves across various plant species. The model is trained to classify 15 different plant disease classes.

## Requirements

Python 3.11 is preferred. The primary dependencies include:

- PyTorch
- torchvision
- matplotlib
- numpy
- Pillow
- tqdm
- pandas
- scikit-learn
- seaborn
- streamlit
- opencv-python

## Project Structure

```
.
├── prepare_dataset.py   # Dataset preparation script
├── train.py             # Model training script 
├── inference.py         # Model inference script for single images
├── evaluate.py          # Model evaluation script for test dataset
├── app.py               # Streamlit web application
├── PlantVillage/        # Original dataset directory
└── PlantVillage_Split/  # Processed dataset with train/val/test splits
    ├── train/           # Training data (70%)
    ├── val/             # Validation data (20%)
    └── test/            # Test data (10%)
```

## Usage

### 1. Dataset Preparation

First, split the dataset into train, validation, and test sets:

```bash
python prepare_dataset.py --source ./PlantVillage --output ./PlantVillage_Split --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

This will create a new directory structure with train, validation, and test splits while preserving the class structure.

### 2. Model Training

Train the model using the split dataset:

```bash
python train.py --model mobilenet_v3_large --batch-size 32 --epochs 20 --lr 0.001
```

Additional training options:

```
--data-dir: Path to the dataset directory (default: ./PlantVillage_Split)
--img-size: Input image size (default: 224)
--batch-size: Batch size for training (default: 32)
--num-workers: Number of workers for data loading (default: 4)
--model: Model architecture (choices: mobilenet_v3_small, mobilenet_v3_large, default: mobilenet_v3_large)
--pretrained: Use pretrained model (default: True)
--freeze-backbone: Freeze backbone layers and train only the classifier (default: False)
--epochs: Number of total epochs to run (default: 20)
--lr: Initial learning rate (default: 0.001)
--weight-decay: Weight decay (default: 1e-4)
--mixed-precision: Use mixed precision training (default: True)
--seed: Random seed (default: 42)
--output-dir: Path to save outputs (default: ./outputs)
```

### 3. Inference on Single Images

Make predictions on individual images:

```bash
python inference.py --image path/to/your/image.jpg
```

Options:

```
--image: Path to the input image (required)
--model-path: Path to the trained model file (default: ./outputs/best_model.pth)
--model-type: Model architecture (choices: mobilenet_v3_small, mobilenet_v3_large, default: mobilenet_v3_large)
--class-names: Path to the class names file (default: ./outputs/class_names.txt)
--img-size: Input image size (default: 224)
```

### 4. Model Evaluation on Test Dataset

Evaluate the model performance on the test dataset:

```bash
python evaluate.py
```

This will:
- Calculate accuracy, precision, recall, and F1 score metrics
- Generate a confusion matrix visualization
- Save all metrics to an Excel file
- Print a detailed classification report to the terminal

Options:

```
--data-dir: Path to test data directory (default: ./PlantVillage_Split/test)
--model-path: Path to the trained model file (default: ./outputs/best_model.pth)
--model-type: Model architecture (default: mobilenet_v3_large)
--class-names: Path to the class names file (default: ./outputs/class_names.txt)
--img-size: Input image size (default: 224)
--batch-size: Batch size for evaluation (default: 64)
--num-workers: Number of workers for data loading (default: 4)
--output-dir: Path to save evaluation results (default: ./evaluation)
```

### 5. Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run app.py
```

Features:
- Upload custom images for classification
- View model predictions with confidence scores
- See GradCAM visualizations showing model focus areas
- Inference time measurement in milliseconds
- Top 5 predictions for each image

## Features

- **Data Augmentation**: The training pipeline includes various data augmentation techniques like random crops, flips, rotations, and color jitter.
- **Mixed Precision Training**: Uses PyTorch's automatic mixed precision for faster training and reduced memory usage.
- **Training Visualization**: Generates plots of training and validation loss/accuracy.
- **Model Checkpointing**: Saves best model based on validation accuracy.
- **Inference Visualization**: Visualizes model predictions with confidence scores.
- **Model Evaluation**: Comprehensive evaluation with confusion matrix and detailed metrics.
- **GradCAM Visualization**: Highlights areas of the image that influenced the model's decision.
- **Streamlit UI**: User-friendly web interface for model interaction.

## Training Details

The model is trained with the following setup:

- **Base Architecture**: MobileNetV3 (large or small variant)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Learning Rate Scheduler**: Cosine Annealing
- **Batch Size**: Default is 32
- **Image Size**: 224x224 pixels (3 channels)
- **Class Balance**: The dataset is split class-wise to maintain class distribution

## Results

After training, the model generates:

1. Training and validation curves showing loss and accuracy
2. Best model checkpoint saved in the output directory
3. Last checkpoint containing all training history

The evaluation script generates:

1. Confusion matrix visualization
2. Excel file with detailed metrics
3. Terminal output with classification report

## Customization

You can customize various aspects of the training:

- Change the model architecture (mobilenet_v3_small vs mobilenet_v3_large)
- Adjust learning rate and optimizer parameters
- Modify training/validation split ratios
- Fine-tune a pre-trained model by freezing the backbone

## License

This project is open-source. Please respect the licenses of the dependencies and the dataset used. 