import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pathlib import Path

from train import ModelBuilder


def load_model(model_path, model_type, num_classes, device):
    """Load a trained model from checkpoint"""
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


def get_transform(size=224):
    """Define transform for inference"""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(model, image_path, transform, class_names, device):
    """Predict class for an image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    predictions = []
    for i, (prob, idx) in enumerate(zip(top5_prob.cpu().numpy(), top5_indices.cpu().numpy())):
        predictions.append({
            'rank': i + 1,
            'class_name': class_names[idx],
            'probability': float(prob)
        })
    
    return predictions


def visualize_prediction(image_path, predictions):
    """Visualize the prediction with a bar chart"""
    # Load and display image
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display prediction bar chart
    plt.subplot(1, 2, 2)
    classes = [p['class_name'] for p in predictions]
    probs = [p['probability'] for p in predictions]
    
    y_pos = np.arange(len(classes))
    
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Top 5 Predictions')
    
    plt.tight_layout()
    
    # Save and show the figure
    output_path = f"prediction_{Path(image_path).stem}.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plant disease prediction')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--model-path', type=str, default='./outputs/best_model.pth',
                        help='Path to the trained model file')
    parser.add_argument('--model-type', type=str, default='mobilenet_v3_large',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Model architecture (default: mobilenet_v3_large)')
    parser.add_argument('--class-names', type=str, default='./outputs/class_names.txt',
                        help='Path to the class names file')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return
    
    # Check if model exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    # Check if class names file exists
    if not os.path.isfile(args.class_names):
        print(f"Error: Class names file '{args.class_names}' not found.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class names
    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)
    
    # Load model
    model = load_model(args.model_path, args.model_type, num_classes, device)
    
    # Get transform
    transform = get_transform(args.img_size)
    
    # Predict
    predictions = predict_image(model, args.image, transform, class_names, device)
    
    # Print predictions
    print("\nPredictions:")
    for pred in predictions:
        print(f"{pred['rank']}. {pred['class_name']}: {pred['probability']:.4f}")
    
    # Visualize
    visualize_prediction(args.image, predictions)


if __name__ == "__main__":
    main() 