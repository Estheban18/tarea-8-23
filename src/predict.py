import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import Tuple, Dict
import json
import os

class ImagePredictor:
    def __init__(self, model_path: str = "models/object_model.h5", class_mapping: str = "models/class_mapping.json"):
        """
        Initialize the image predictor with model and class mappings
        
        Args:
            model_path: Path to the trained Keras model
            class_mapping: Path to JSON file containing class index to label mapping
        """
        self.model = self._load_model(model_path)
        self.class_mapping = self._load_class_mapping(class_mapping)
        self.input_size = (64, 64)  # Should match model's expected input shape
        
    def _load_model(self, model_path: str):
        """Safely load Keras model with error handling"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return load_model(model_path)
    
    def _load_class_mapping(self, mapping_path: str) -> Dict:
        """Load class index to label mapping"""
        try:
            with open(mapping_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Class mapping file not found at {mapping_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in mapping file at {mapping_path}")
            return {}

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, self.input_size)
        img = img.astype('float32') / 255.0  # Normalize
        return np.expand_dims(img, axis=0)

    def predict(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Make prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (predicted_label, confidence, all_predictions)
        """
        try:
            processed_img = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_img)
            class_index = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            predicted_label = self.class_mapping.get(str(class_index), f"Class_{class_index}")
            
            # Get all class probabilities
            all_predictions = {
                self.class_mapping.get(str(i), f"Class_{i}"): float(predictions[0][i])
                for i in range(len(predictions[0]))
            }
            
            return predicted_label, confidence, all_predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_and_display(self, image_path: str):
        """
        Make prediction and display results with visualization
        
        Args:
            image_path: Path to the image file
        """
        try:
            predicted_label, confidence, all_predictions = self.predict(image_path)
            
            # Display results
            print("\n" + "="*50)
            print(f"Prediction Results for {os.path.basename(image_path)}:")
            print("-"*50)
            print(f"Predicted class: {predicted_label}")
            print(f"Confidence: {confidence:.2%}")
            print("\nAll class probabilities:")
            for label, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                print(f"{label}: {prob:.2%}")
            print("="*50 + "\n")
            
            # Visualization
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Add prediction text to image
            text = f"{predicted_label} ({confidence:.1%})"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display image (requires GUI environment)
            cv2.imshow("Prediction Result", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    predictor = ImagePredictor()
    
    # Single prediction with full output
    label, confidence, _ = predictor.predict("test_image.jpg")
    print(f"Predicted: {label} with {confidence:.2%} confidence")
    
    # Full prediction with visualization
    predictor.predict_and_display("test_image.jpg")