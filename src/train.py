import os
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import (
ModelCheckpoint,
EarlyStopping,
ReduceLROnPlateau,
TensorBoard
)
from src.dataset_loader import load_data
from src.model import create_model
from src.utils import setup_gpu_memory_growth, create_dir_if_not_exists


class ModelTrainer:
def __init__(self, config: dict):
"""
Initialize the model trainer with configuration

Args:
config (dict): Training configuration dictionary containing:
- data_path: Path to training data
- input_shape: Model input shape (height, width, channels)
- batch_size: Training batch size
- epochs: Maximum number of epochs
- validation_split: Ratio for validation split
- model_dir: Directory to save models
- log_dir: Directory for TensorBoard logs
- patience: Early stopping patience
"""
self.config = config
setup_gpu_memory_growth()
self._setup_directories()

def _setup_directories(self):
"""Create necessary directories if they don't exist"""
create_dir_if_not_exists(self.config['model_dir'])
create_dir_if_not_exists(self.config['log_dir'])

def _get_callbacks(self) -> list:
"""Configure training callbacks"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
ModelCheckpoint(
filepath=os.path.join(self.config['model_dir'], 'best_model.h5'),
monitor='val_accuracy',
save_best_only=True,
mode='max',
verbose=1
),
EarlyStopping(
monitor='val_loss',
patience=self.config['patience'],
restore_best_weights=True
),
ReduceLROnPlateau(
monitor='val_loss',
factor=0.1,
patience=self.config['patience'] // 2
),
TensorBoard(
log_dir=os.path.join(self.config['log_dir'], timestamp),
histogram_freq=1
)
]
return callbacks

def train(self):
"""Execute the full training pipeline"""
try:
# Load and prepare data
print("\n⏳ Loading training data...")
X_train, y_train, classes = load_data(self.config['data_path'])

# Data validation
self._validate_data(X_train, y_train, classes)

# Create model
print("\n🛠️ Creating model architecture...")
model = create_model(self.config['input_shape'], len(classes))

# Display model summary
model.summary()

# Train model
print("\n🚀 Starting model training...")
history = model.fit(
X_train,
y_train,
batch_size=self.config['batch_size'],
epochs=self.config['epochs'],
validation_split=self.config['validation_split'],
callbacks=self._get_callbacks(),
verbose=2
)

# Save final model
final_model_path = os.path.join(
self.config['model_dir'],
f"final_model_{datetime.now().strftime('%Y%m%d')}.h5"
)
model.save(final_model_path)
print(f"\n✅ Training complete! Model saved to {final_model_path}")

return history

except Exception as e:
print(f"\n❌ Training failed: {str(e)}")
raise

def _validate_data(self, X_train: np.ndarray, y_train: np.ndarray, classes: list):
"""Validate training data before training"""
if len(X_train) != len(y_train):
raise ValueError("Mismatch between number of samples and labels")

if len(classes) != y_train.shape[1]:
raise ValueError("Number of classes doesn't match output shape")

print(f"\n📊 Dataset Info:")
print(f"- Training samples: {len(X_train)}")
print(f"- Number of classes: {len(classes)}")
print(f"- Input shape: {X_train[0].shape}")
print(f"- Classes: {', '.join(classes)}")


if __name__ == "__main__":
# Configuration dictionary
config = {
'data_path': "data/train",
'input_shape': (64, 64, 3),
'batch_size': 32,
'epochs': 50,
'validation_split': 0.2,
'model_dir': "models",
'log_dir': "logs",
'patience': 10
}

# Execute training
trainer = ModelTrainer(config)
history = trainer.train()   
