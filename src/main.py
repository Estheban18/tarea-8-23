import os
import cv2
import numpy as np
from src.predict import predict_image
import matplotlib.pyplot as plt
import time

def main():
    # Configuration
    config = {
        'image_path': "data/test/sample.jpg",  # Change this to your image path
        'model_path': "models/object_model.h5",  # Optional: if not already loaded in predict.py
        'class_names': ["class1", "class2", "class3"],  # Replace with your actual class names
        'show_confidence': True,
        'save_output': True,
        'output_dir': "results/predictions"
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    try:
        # Load and process image
        if not os.path.exists(config['image_path']):
            raise FileNotFoundError(f"Image not found at {config['image_path']}")

        print(f"\nPredicting image: {config['image_path']}")
        start_time = time.time()

        # Make prediction
        prediction = predict_image(
            image_path=config['image_path'],
            model_path=config.get('model_path'),
            class_names=config.get('class_names')
        )

        # Display results
        print(f"\nPrediction completed in {time.time()-start_time:.2f} seconds")
        print("\nPrediction Results:")
        print(f"Predicted Class: {prediction['class_name']}")
        print(f"Class ID: {prediction['class_id']}")
        if config['show_confidence']:
            print(f"Confidence: {prediction['confidence']:.2%}")
        
        # Visualize the image with prediction
        img = cv2.cvtColor(cv2.imread(config['image_path']), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        title = f"Prediction: {prediction['class_name']}"
        if config['show_confidence']:
            title += f" ({prediction['confidence']:.2%})"
        plt.title(title)
        plt.axis('off')
        
        # Save or show results
        if config['save_output']:
            output_path = os.path.join(
                config['output_dir'],
                f"prediction_{os.path.basename(config['image_path'])}"
            )
            plt.savefig(output_path)
            print(f"\nPrediction visualization saved to: {output_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        if isinstance(e, FileNotFoundError):
            print("Please check the image path and try again.")
        elif "predict_image" in str(e):
            print("There was an error in the prediction function.")
        else:
            print("An unexpected error occurred.")

if __name__ == "__main__":
    main()