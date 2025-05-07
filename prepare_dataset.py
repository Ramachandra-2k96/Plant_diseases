import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train, validation and test sets while preserving class structure.
    
    Args:
        source_dir: Source directory containing class folders
        output_dir: Output directory where train, val, test folders will be created
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.2)
        test_ratio: Ratio of test data (default: 0.1)
    """
    # Validate ratios
    if not isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each class directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        print(f"Processing class: {class_name}")
        
        # Create class directories in train, val and test
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all files in the class directory
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(files)  # Randomize files
        
        # Calculate split sizes
        num_files = len(files)
        num_train = int(train_ratio * num_files)
        num_val = int(val_ratio * num_files)
        # num_test will be the remainder to ensure we use all files
        
        # Split files
        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]
        
        print(f"  Total: {num_files}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copy files to respective directories
        for file_name in train_files:
            shutil.copy2(
                os.path.join(class_dir, file_name),
                os.path.join(train_class_dir, file_name)
            )
        
        for file_name in val_files:
            shutil.copy2(
                os.path.join(class_dir, file_name),
                os.path.join(val_class_dir, file_name)
            )
        
        for file_name in test_files:
            shutil.copy2(
                os.path.join(class_dir, file_name),
                os.path.join(test_class_dir, file_name)
            )

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train, val, and test sets')
    parser.add_argument('--source', type=str, default='./PlantVillage', 
                        help='Source directory containing class folders')
    parser.add_argument('--output', type=str, default='./PlantVillage_Split',
                        help='Output directory for split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')  
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of test data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Splitting dataset with ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    split_dataset(
        args.source, 
        args.output, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    print("Dataset split completed successfully!") 