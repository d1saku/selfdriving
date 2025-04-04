import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os
import sys
import cv2
from pathlib import Path

# Add project root to path (considering utils folder location)
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import model and dataset classes
from src.model.model import PNetCNN, build_model
from src.preprocessing.augmentation import DrivingDataset, create_dataloaders
from src.training.train import DrivingModel

# Paths (relative to project root)
LABELS_VAL_PATH = os.path.join(project_root, "data/validation.csv")
TRAINING_PATH = os.path.join(project_root, "data/training_data/")
MODEL_PATH = os.path.join(project_root, "models/best_model.h5")
LOGS_PATH = os.path.join(project_root, "training_log.pkl")

def load_training_log(log_path=LOGS_PATH):
    """Load training log data from pickle file"""
    if not os.path.exists(log_path):
        print(f"Training log not found at {log_path}")
        return None
    
    with open(log_path, 'rb') as f:
        return pickle.load(f)

def plot_training_history(log_data, save_path=None):
    """Plot training and validation loss history"""
    if log_data is None:
        return
    
    plt.figure(figsize=(12, 10))
    
    # Plot combined losses
    plt.subplot(2, 2, 1)
    plt.plot(log_data['training'], label='Training Loss')
    plt.plot(log_data['validation'], label='Validation Loss')
    plt.title('Combined Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot angle losses
    plt.subplot(2, 2, 2)
    plt.plot(log_data['angle_loss'], label='Angle Loss')
    plt.plot(log_data['angle_vloss'], label='Validation Angle Loss')
    plt.title('Steering Angle Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot speed losses
    plt.subplot(2, 2, 3)
    plt.plot(log_data['speed_loss'], label='Speed Loss')
    plt.plot(log_data['speed_vloss'], label='Validation Speed Loss')
    plt.title('Speed Loss')
    plt.legend()
    plt.grid(True)
    
    # Log scale plot for all losses
    plt.subplot(2, 2, 4)
    plt.semilogy(log_data['training'], label='Training Loss')
    plt.semilogy(log_data['validation'], label='Validation Loss')
    plt.semilogy(log_data['angle_loss'], label='Angle Loss')
    plt.semilogy(log_data['angle_vloss'], label='Val Angle Loss')
    plt.semilogy(log_data['speed_loss'], label='Speed Loss')
    plt.semilogy(log_data['speed_vloss'], label='Val Speed Loss')
    plt.title('All Losses (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_model(model_path=MODEL_PATH):
    """Load a trained model from weights file"""
    # Create model architecture first
    input_shape = (66, 200, 3)  # Should match training
    base_model = build_model(input_shape=input_shape)
    model = DrivingModel(base_model)
    
    # Compile with same loss functions as training
    model.compile(
        optimizer='adam',  # Doesn't matter for inference
        loss={
            'angle': tf.keras.losses.MeanSquaredError(),
            'speed': tf.keras.losses.BinaryCrossentropy()
        },
        metrics=['mae']
    )
    
    # Load weights
    try:
        model.load_weights(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def visualize_predictions(model, num_samples=10, save_dir=None):
    """Visualize model predictions on validation data"""
    if model is None:
        return

    # Create validation dataset
    _, val_dataset = create_dataloaders(
        train_csv=None,  # Not needed for validation only
        val_csv=LABELS_VAL_PATH,
        img_dir=TRAINING_PATH,
        batch_size=num_samples
    )
    
    # Get a batch of data
    for images, labels in val_dataset.take(1):
        break
    
    # Make predictions
    predictions = model(images, training=False)
    predicted_angles = tf.squeeze(predictions['angle']).numpy()
    predicted_speeds = tf.squeeze(predictions['speed']).numpy()
    
    true_angles = labels[:, 0].numpy()
    true_speeds = labels[:, 1].numpy()
    
    # Create a grid of images with predictions
    fig = plt.figure(figsize=(15, num_samples * 3))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(num_samples, 2, i*2 + 1)
        img = images[i].numpy()
        # Denormalize if necessary (assuming [0,1] range)
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"True: Angle={true_angles[i]:.2f}, Speed={true_speeds[i]:.2f}\n"
                  f"Pred: Angle={predicted_angles[i]:.2f}, Speed={predicted_speeds[i]:.2f}")
        plt.axis('off')
        
        # Visualize steering angle
        ax = plt.subplot(num_samples, 2, i*2 + 2)
        draw_steering_wheel(ax, predicted_angles[i], true_angles[i])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "predictions.png"))
    
    plt.show()

def draw_steering_wheel(ax, predicted_angle, true_angle, radius=1.0):
    """Draw a steering wheel with true and predicted angles"""
    # Draw the steering wheel circle
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2)
    ax.add_artist(circle)
    
    # Draw the center marker
    ax.plot([0, 0], [-0.1, 0.1], 'k-', linewidth=2)
    ax.plot([-0.1, 0.1], [0, 0], 'k-', linewidth=2)
    
    # Convert angles to radians (assuming angles are in -1 to 1 range, scaled to -π/4 to π/4)
    true_rad = true_angle * np.pi / 4
    pred_rad = predicted_angle * np.pi / 4
    
    # Draw true angle line
    ax.plot([0, radius * np.sin(true_rad)], [0, radius * np.cos(true_rad)], 'g-', linewidth=3, label='True')
    
    # Draw predicted angle line
    ax.plot([0, radius * np.sin(pred_rad)], [0, radius * np.cos(pred_rad)], 'r-', linewidth=3, label='Predicted')
    
    # Configure the plot
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('Steering Angle')
    ax.set_axis_off()

def visualize_feature_maps(model, image_path, layer_name=None, save_dir=None):
    """Visualize activation maps of a specific layer given an input image"""
    if model is None:
        return
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))  # Resize to model input size
    img = img / 255.0  # Normalize to [0,1]
    img_tensor = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # If no layer specified, use the first conv layer
    if layer_name is None:
        for layer in model.base_model.layers:
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    # Create a feature extraction model
    feature_model = tf.keras.Model(
        inputs=model.base_model.input,
        outputs=model.base_model.get_layer(layer_name).output
    )
    
    # Get feature maps
    feature_maps = feature_model.predict(img_tensor)
    
    # Plot original image and feature maps
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.subplot(4, 4, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Show feature maps
    n_features = min(15, feature_maps.shape[-1])
    for i in range(n_features):
        plt.subplot(4, 4, i + 2)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.title(f"Feature {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"feature_maps_{layer_name}.png"))
    
    plt.show()

def main():
    # Create output directory for saving visualizations
    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualize training history
    log_data = load_training_log()
    if log_data:
        plot_training_history(log_data, save_path=os.path.join(output_dir, "training_history.png"))
    
    # 2. Load model
    model = load_model()
    
    # 3. Visualize predictions
    if model:
        visualize_predictions(model, num_samples=5, save_dir=output_dir)
    
    # 4. Visualize feature maps (if there's a sample image)
    # Find first image in validation data
    try:
        from glob import glob
        sample_images = glob(os.path.join(TRAINING_PATH, "*.jpg"))
        if sample_images:
            visualize_feature_maps(model, sample_images[0], save_dir=output_dir)
    except Exception as e:
        print(f"Could not visualize feature maps: {e}")

if __name__ == "__main__":
    main()