import os
import time
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import cv2
from pathlib import Path
import io

from train import ModelBuilder

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        x.requires_grad_()
        
        # Get model output
        output = self.model(x)
        logits = output
        
        # If class_idx is None, get the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        # Target for backprop
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights from gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Get weighted activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.squeeze().cpu().numpy()

def get_transform(size=224):
    """Define transform for inference"""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

def predict_with_gradcam(model, image, transform, class_names, device):
    """Make prediction and generate GradCAM visualization"""
    # Prepare image tensor
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Calculate inference time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Inference time in milliseconds
    inference_time = (time.time() - start_time) * 1000
    
    # Get prediction
    pred_idx = torch.argmax(probabilities).item()
    pred_class = class_names[pred_idx]
    confidence = probabilities[pred_idx].item()
    
    # Get GradCAM
    # For MobileNetV3, get the last convolutional layer
    if hasattr(model, 'features'):
        target_layer = model.features[-1]
    else:
        # Fallback for other models
        target_layer = list(model.modules())[-3]
    
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(image_tensor, pred_idx)
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy array for overlay
    img_array = np.array(image.convert('RGB').resize((224, 224)))
    
    # Create overlay
    overlay = heatmap * 0.4 + img_array * 0.6
    overlay = np.uint8(overlay)
    
    # Get top 5 predictions
    top5_values, top5_indices = torch.topk(probabilities, 5)
    top5_predictions = [(class_names[idx], val.item()) for idx, val in zip(top5_indices, top5_values)]
    
    return {
        'pred_class': pred_class,
        'confidence': confidence,
        'inference_time': inference_time,
        'gradcam': overlay,
        'top5_predictions': top5_predictions
    }

def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
    
    st.title("Plant Disease Classification")
    st.write("Upload an image to classify plant diseases using a MobileNetV3 model")
    
    # Sidebar for model selection
    st.sidebar.header("Model Settings")
    model_type = st.sidebar.selectbox(
        "Select model type",
        ["mobilenet_v3_large", "mobilenet_v3_small"]
    )
    
    model_path = st.sidebar.text_input(
        "Model path",
        value="./outputs/best_model.pth"
    )
    
    class_names_path = st.sidebar.text_input(
        "Class names path",
        value="./outputs/class_names.txt"
    )
    
    # Check if files exist
    if not os.path.isfile(model_path):
        st.sidebar.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.isfile(class_names_path):
        st.sidebar.error(f"Class names file not found: {class_names_path}")
        return
    
    # Load model and class names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Using device: {device}")
    
    # Load class names
    try:
        class_names = load_class_names(class_names_path)
        num_classes = len(class_names)
        st.sidebar.success(f"Loaded {num_classes} classes")
    except Exception as e:
        st.sidebar.error(f"Failed to load class names: {str(e)}")
        return
    
    # Load model (only when needed)
    @st.cache_resource
    def get_model(_model_path, _model_type, _num_classes, _device):
        return load_model(_model_path, _model_type, _num_classes, _device)
    
    try:
        model = get_model(model_path, model_type, num_classes, device)
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")
        return
    
    # Get transform
    transform = get_transform()
    
    # Simple file uploader without type restrictions
    uploaded_file = st.file_uploader("Choose an image...")
    
    if uploaded_file is not None:
        # Check if the file is a valid image type
        if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Display original image
                image = Image.open(uploaded_file).convert("RGB")
                
                # Make prediction with GradCAM
                results = predict_with_gradcam(model, image, transform, class_names, device)
                
                # Show results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("Original Image")
                    st.image(image, use_container_width=True)
                    st.write(f"**Prediction:** {results['pred_class']}")
                    st.write(f"**Confidence:** {results['confidence']:.4f}")
                    st.write(f"**Inference Time:** {results['inference_time']:.2f} ms")
                    
                    # Display top 5 predictions
                    st.subheader("Top 5 Predictions")
                    for i, (class_name, prob) in enumerate(results['top5_predictions']):
                        st.write(f"{i+1}. {class_name}: {prob:.4f}")
                
                with col2:
                    st.header("GradCAM Visualization")
                    st.image(results['gradcam'], use_container_width=True)
                    st.write("The highlighted areas show the regions the model focused on to make its prediction.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please make sure you've uploaded a valid image file.")
        else:
            st.error("Please upload an image file (jpg, jpeg, or png).")
    else:
        st.info("Please upload an image to get a prediction.")

if __name__ == "__main__":
    main() 