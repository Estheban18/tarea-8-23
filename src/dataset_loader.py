import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(64, 64), classes=None):
    """
    Load image data from directory structure where each subdirectory represents a class.
    
    Args:
        data_dir (str): Path to the root directory containing class subdirectories
        img_size (tuple): Target size for resizing images (width, height)
        classes (list): Optional list of class names to include. If None, uses all subdirectories.
    
    Returns:
        tuple: (X, y, classes) where:
            X: numpy array of preprocessed images
            y: one-hot encoded class labels
            classes: list of class names in order they were processed
    """
    X, y = [], []
    
    # Validate input directory
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Get class names
    if classes is None:
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes = sorted(classes)  # Ensure consistent ordering
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Class directory not found, skipping: {class_path}")
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                # Convert BGR to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                X.append(img)
                y.append(label)
                
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {str(e)}")
                continue
    
    if not X:
        raise ValueError("No valid images found in the dataset")
    
    # Convert to numpy arrays and normalize
    X = np.array(X, dtype=np.float32) / 255.0
    y = to_categorical(np.array(y), num_classes=len(classes))
    
    return X, y, classes