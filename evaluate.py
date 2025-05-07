import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from train import PlantDataset, ModelBuilder

def get_transform(size=224):
    """Define transform for evaluation"""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_model(model_path, model_type, num_classes, device):
    """Load a trained model from checkpoint"""
    # Set pretrained=False to avoid warning, as we're loading our own weights
    model = ModelBuilder.build_model(model_type, num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def load_class_names(class_names_path):
    """Load class names from file"""
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(16, 12))
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def evaluate_model(model, data_loader, device, class_names):
    """Evaluate the model on the test set"""
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Get detailed classification report
    class_report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    return {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'predictions': all_preds,
        'targets': all_targets
    }

def save_to_excel(results, class_names, output_path):
    """Save evaluation results to Excel"""
    # Create a writer to save multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary Metrics
        summary_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Per-Class Metrics
        class_report = results['classification_report']
        class_metrics = []
        
        for i, class_name in enumerate(class_names):
            class_metrics.append({
                'Class': class_name,
                'Precision': class_report[class_name]['precision'],
                'Recall': class_report[class_name]['recall'],
                'F1-Score': class_report[class_name]['f1-score'],
                'Support': int(class_report[class_name]['support'])
            })
        
        pd.DataFrame(class_metrics).to_excel(writer, sheet_name='Class Metrics', index=False)
        
        # Sheet 3: Confusion Matrix
        conf_matrix_df = pd.DataFrame(results['confusion_matrix'], 
                                      columns=class_names, 
                                      index=class_names)
        conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate plant disease classification model on test set')
    
    parser.add_argument('--data-dir', type=str, default='./PlantVillage_Split/test',
                        help='Path to test data directory')
    parser.add_argument('--model-path', type=str, default='./outputs/best_model.pth',
                        help='Path to the trained model file')
    parser.add_argument('--model-type', type=str, default='mobilenet_v3_large',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Model architecture (default: mobilenet_v3_large)')
    parser.add_argument('--class-names', type=str, default='./outputs/class_names.txt',
                        help='Path to the class names file')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                        help='Path to save evaluation results (default: ./evaluation)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    # Check if class names file exists
    if not os.path.isfile(args.class_names):
        print(f"Error: Class names file '{args.class_names}' not found.")
        return
    
    # Load class names
    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes")
    
    # Create test dataset and dataloader
    test_transform = get_transform(args.img_size)
    test_dataset = PlantDataset(root_dir=args.data_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model = load_model(args.model_path, args.model_type, num_classes, device)
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print metrics to terminal
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(results['targets'], results['predictions'], target_names=class_names))
    
    # Plot confusion matrix
    conf_matrix_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(results['confusion_matrix'], class_names, conf_matrix_path)
    
    # Save results to Excel
    excel_path = output_dir / 'evaluation_results.xlsx'
    save_to_excel(results, class_names, excel_path)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main() 