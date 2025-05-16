import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torch.amp import autocast, GradScaler
import argparse
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import datetime
import collections


class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self.targets = []  # Add a list to store targets for easier sampling weight calculation
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    class_idx = self.class_to_idx[class_name]
                    self.samples.append((img_path, class_idx))
                    self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ModelBuilder:
    @staticmethod
    def build_model(model_name, num_classes, pretrained=True):
        """
        Build and return the model based on the specified architecture
        """
        if model_name == 'mobilenet_v3_small':
            if pretrained:
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif model_name == 'mobilenet_v3_large':
            if pretrained:
                model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            else:
                model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        return model


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, 
                 device, epochs, mixed_precision=True, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.mixed_precision = mixed_precision
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize lists to store metrics for plotting
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # For mixed precision training
        self.scaler = GradScaler('cuda') if mixed_precision else None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Scale gradients and perform backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Compute metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (pbar.n + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}', 
                'acc': f'{acc:.2f}%'
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        return train_loss, train_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Compute metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                avg_loss = running_loss / (pbar.n + 1)
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}', 
                    'acc': f'{acc:.2f}%'
                })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        return val_loss, val_acc
    
    def train(self):
        """Main training loop"""
        print(f"Training on {self.device}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }, self.save_dir / 'best_model.pth')
                print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            
            # Save checkpoint at the end of each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs,
            }, self.save_dir / f'last_checkpoint.pth')
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.train_losses, self.train_accs, self.val_losses, self.val_accs
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        epochs_range = range(1, len(self.train_losses) + 1)
        plt.plot(epochs_range, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs_range, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs_range, self.val_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()


def get_transforms(size=224):
    """Define transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def get_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance"""
    # Count samples per class
    class_counts = collections.Counter(dataset.targets)
    num_classes = len(class_counts)
    
    # Compute weights for each sample
    class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
    sample_weights = [class_weights[target] for target in dataset.targets]
    
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Class distribution: {class_counts}")
    print(f"Applied weights to balance {num_classes} classes")
    
    return sampler


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a plant disease classification model')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./Data',
                        help='Path to the dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data loading (default: 4)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='mobilenet_v3_large',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large'],
                        help='Model architecture (default: mobilenet_v3_large)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained model')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze backbone layers and train only the classifier')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of total epochs to run (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=108,
                        help='Random seed (default: 108)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Path to save outputs (default: ./outputs)')
    parser.add_argument('--weighted-sampling', action='store_true', default=True,
                        help='Use weighted sampling to handle class imbalance')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transforms
    train_transform, val_transform = get_transforms(args.img_size)
    
    # Create datasets
    train_dataset = PlantDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = PlantDataset(
        root_dir=os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create data loaders
    if args.weighted_sampling:
        # Use weighted sampling for training
        train_sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=train_sampler,  # Use our weighted sampler
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model
    model = ModelBuilder.build_model(args.model, num_classes, args.pretrained)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        mixed_precision=args.mixed_precision,
        save_dir=output_dir
    )
    
    # Train the model
    train_losses, train_accs, val_losses, val_accs = trainer.train()
    
    # Save class names
    with open(output_dir / 'class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 