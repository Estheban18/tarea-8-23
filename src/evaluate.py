import os
import time
from src.dataset_loader import load_data
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model_path, test_data_dir, class_names=None, show_samples=False):
    """
    Evaluate a trained model on test data with comprehensive reporting.
    
    Args:
        model_path (str): Path to the saved model file
        test_data_dir (str): Directory containing test data
        class_names (list): Optional list of class names for labeling
        show_samples (bool): Whether to display sample predictions
    
    Returns:
        dict: Evaluation metrics and results
    """
    # Load test data
    try:
        print(f"Loading test data from {test_data_dir}...")
        start_time = time.time()
        X_test, y_test, loaded_classes = load_data(test_data_dir)
        if class_names is None:
            class_names = loaded_classes
        print(f"Loaded {len(X_test)} test samples in {time.time()-start_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {str(e)}")

    # Load model
    try:
        print(f"\nLoading model from {model_path}...")
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

    # Evaluate model
    print("\nEvaluating model...")
    start_time = time.time()
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    evaluation_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

    # Sample predictions visualization
    if show_samples:
        visualize_sample_predictions(X_test, y_true_classes, y_pred_classes, class_names)

    return {
        'loss': loss,
        'accuracy': acc,
        'evaluation_time': evaluation_time,
        'classification_report': classification_report(
            y_true_classes, y_pred_classes, target_names=class_names, output_dict=True),
        'confusion_matrix': cm
    }

def visualize_sample_predictions(images, true_labels, pred_labels, class_names, num_samples=5):
    """Display sample images with their true and predicted labels."""
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(range(len(images)), size=num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[idx])
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png')
    plt.show()

if __name__ == "__main__":
    # Configuration
    config = {
        'model_path': "models/object_model.h5",
        'test_data_dir': "data/test",
        'class_names': None,  # Will be loaded from data if None
        'show_samples': True
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run evaluation
    try:
        metrics = evaluate_model(**config)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")