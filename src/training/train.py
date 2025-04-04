import time
import pickle
import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Import your TensorFlow model and dataset classes
from src.model.model import PNetCNN, build_model
from src.preprocessing.augmentation import DrivingDataset, create_dataloaders

# Paths
LABELS_TRAINING_PATH = "data/train.csv"
LABELS_VAL_PATH = "data/validation.csv"
TRAINING_PATH = "data/training_data/"


class DrivingModel(tf.keras.Model):
    """Custom model wrapper to provide more control over training"""
    def __init__(self, base_model):
        super(DrivingModel, self).__init__()
        self.base_model = base_model
        
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        images, true_labels = data
        
        # Apply epsilon clamping as in PyTorch version
        eps = 1e-7
        true_labels = tf.clip_by_value(true_labels, eps, 1 - eps)
        
        with tf.GradientTape() as tape:
            outputs = self(images, training=True)
            
            angle = tf.squeeze(outputs['angle'])
            speed = tf.squeeze(outputs['speed'])
            
            # Get loss functions from the compiled model
            angle_loss_fn = self.loss['angle'] if isinstance(self.loss, dict) else self.loss
            speed_loss_fn = self.loss['speed'] if isinstance(self.loss, dict) else self.loss
            
            angle_loss = angle_loss_fn(true_labels[:, 0], angle)
            speed_loss = speed_loss_fn(true_labels[:, 1], speed)
            
            loss = angle_loss + speed_loss
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(
            {
                'angle': true_labels[:, 0],
                'speed': true_labels[:, 1]
            },
            {
                'angle': angle,
                'speed': speed
            }
        )
        
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'angle_loss': angle_loss, 'speed_loss': speed_loss})
        return results
    
    def test_step(self, data):
        images, true_labels = data
        
        outputs = self(images, training=False)
        
        angle = tf.squeeze(outputs['angle'])
        speed = tf.squeeze(outputs['speed'])
        
        # Get loss functions from the compiled model
        angle_loss_fn = self.loss['angle'] if isinstance(self.loss, dict) else self.loss
        speed_loss_fn = self.loss['speed'] if isinstance(self.loss, dict) else self.loss
        
        angle_loss = angle_loss_fn(true_labels[:, 0], angle)
        speed_loss = speed_loss_fn(true_labels[:, 1], speed)
        
        loss = angle_loss + speed_loss
        
        # Update metrics
        self.compiled_metrics.update_state(
            {
                'angle': true_labels[:, 0],
                'speed': true_labels[:, 1]
            },
            {
                'angle': angle,
                'speed': speed
            }
        )
        
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'angle_loss': angle_loss, 'speed_loss': speed_loss, 'loss': loss})
        return results


class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    """Custom implementation of ReduceLROnPlateau to match PyTorch behavior"""
    def __init__(self, monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=True):
        super(CustomReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = float('inf')
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose:
                        print(f"\nEpoch {epoch+1}: ReduceLROnPlateau reducing learning "
                              f"rate to {new_lr}.")
                    self.wait = 0


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback to save the best model based on validation loss"""
    def __init__(self, model_path='models/best_model'):
        super(BestModelCheckpoint, self).__init__()
        self.model_path = model_path
        self.best_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            
            # Save the model for the current epoch
            epoch_path = os.path.join('models', f'epoch_{epoch+1}.h5')
            self.model.save_weights(epoch_path)
            print(f'Model saved at {epoch_path}')
            
            # Also save as the best model
            self.model.save_weights(f'{self.model_path}.h5')
            
            # Save in TensorFlow SavedModel format
            self.model.save(f'{self.model_path}_saved_model')
            print(f'Best model saved at {self.model_path}')


def train_model(num_epochs=20, batch_size=64):
    """
    Train the model for specified number of epochs
    
    Args:
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
    """
    
    # Create the datasets
    train_dataset, val_dataset = create_dataloaders(
        train_csv=LABELS_TRAINING_PATH,
        val_csv=LABELS_VAL_PATH,
        img_dir=TRAINING_PATH,
        batch_size=batch_size
    )
    
    # Create the base model
    input_shape = (66, 200, 3)  # Adjust according to your image dimensions
    base_model = build_model(input_shape=input_shape)
    
    # Wrap with custom model for more control
    model = DrivingModel(base_model)
    
    # Define loss functions
    angle_loss = tf.keras.losses.MeanSquaredError()
    speed_loss = tf.keras.losses.BinaryCrossentropy()
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-4)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss={'angle': angle_loss, 'speed': speed_loss},
        metrics=['mae']
    )
    
    # Define callbacks
    callbacks = [
        CustomReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=True),
        BestModelCheckpoint(),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='epoch'),
        # Save training log as pickle
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_training_log(logs, epoch)
        )
    ]
    
    # Training loop
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f'Total training time: {time.time() - start_time:.2f} seconds')
    
    # Save the final model
    model.save_weights('models/final_model.h5')
    model.save('models/final_model_saved_model')
    
    return model, history


def save_training_log(logs, epoch):
    """Save the training logs to a pickle file"""
    # If the file exists, load the existing log
    if os.path.exists('training_log.pkl'):
        with open('training_log.pkl', 'rb') as f:
            training_log = pickle.load(f)
    else:
        # Otherwise create a new log dictionary
        training_log = {
            'training': [],
            'angle_loss': [],
            'speed_loss': [],
            'validation': [],
            'angle_vloss': [],
            'speed_vloss': []
        }
    
    # Update the log with current epoch data
    training_log['training'].append(logs.get('loss'))
    training_log['angle_loss'].append(logs.get('angle_loss'))
    training_log['speed_loss'].append(logs.get('speed_loss'))
    training_log['validation'].append(logs.get('val_loss'))
    training_log['angle_vloss'].append(logs.get('val_angle_loss'))
    training_log['speed_vloss'].append(logs.get('val_speed_loss'))
    
    # Save the updated log
    with open('training_log.pkl', 'wb') as f:
        pickle.dump(training_log, f)


if __name__ == "__main__":
    # Set memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Train the model
    model, history = train_model(num_epochs=20, batch_size=64)
    
    print("Training completed successfully!")