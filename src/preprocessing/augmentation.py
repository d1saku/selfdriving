import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


class DrivingDataset:
    def __init__(self, csv_path, img_dir, batch_size=32, transform=None, augment=False, is_train=True, shuffle=True):
        """
        Dataset for self-driving car with angle and speed prediction
        
        Args:
            csv_path: Path to CSV file with labels
            img_dir: Directory containing images
            batch_size: Batch size for the dataset
            transform: Custom transformation function to apply
            augment: Whether to apply data augmentation
            is_train: Whether this is a training dataset (affects augmentation)
            shuffle: Whether to shuffle the dataset
        """
        self.img_labels = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.custom_transform = transform
        self.augment = augment
        self.is_train = is_train
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.img_labels)
    
    def _parse_function(self, image_id, angle, speed):
        """Parse function for tf.data pipeline"""
        # Convert image_id to string and add .png extension
        img_path = tf.strings.join([self.img_dir, '/', tf.strings.as_string(image_id), '.png'])
        
        # Read the image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=3)

        # In the _parse_function method, after decoding the image:
        img = tf.image.resize(img, [66, 200])
        
        # Convert to float and normalize
        img = tf.cast(img, tf.float32) / 255.0
        
        # Binarize the speed: 0 if it's exactly 0, otherwise 1
        speed_binary = tf.cast(tf.greater(speed, 0), tf.float32)
        
        # Create label tensor
        label = tf.stack([angle, speed_binary])
        
        return img, label
    
    def _augment_function(self, image, label):
        """Apply data augmentation to the image"""
        if not self.augment or not self.is_train:
            return image, label
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        # Random hue
        image = tf.image.random_hue(image, max_delta=0.05)
        
        # Random affine-like transformations
        # 1. Small rotation
        angle = tf.random.uniform([], -5.0, 5.0, dtype=tf.float32)
        image = tf.image.rot90(image, k=tf.cast(angle / 90.0, tf.int32))
        
        # 2. Small translation is handled by random crop with padding
        # Add padding
        paddings = tf.constant([[5, 5], [5, 5], [0, 0]])
        image = tf.pad(image, paddings, "CONSTANT")
        
        # Random crop back to original size
        original_shape = tf.shape(image)
        image = tf.image.random_crop(image, [original_shape[0]-10, original_shape[1]-10, 3])
        
        # 3. Random scale is more complex in tf.data pipeline and might require custom ops
        
        # Apply custom transform if provided
        if self.custom_transform is not None:
            image = self.custom_transform(image)
        
        return image, label
    
    def create_dataset(self):
        """Create a tf.data.Dataset from the dataframe"""
        # Create tensors for image IDs and labels
        image_ids = self.img_labels['image_id'].values
        angles = self.img_labels['angle'].values.astype(np.float32)
        speeds = self.img_labels['speed'].values.astype(np.float32)
        
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((image_ids, angles, speeds))
        
        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.img_labels))
        
        # Map functions to parse images and labels
        dataset = dataset.map(self._parse_function, 
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Apply augmentation
        dataset = dataset.map(self._augment_function,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Batch the dataset
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return dataset


# Helper function to create datasets
def create_dataloaders(train_csv, val_csv, img_dir, batch_size=32):
    """
    Create train and validation datasets
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        img_dir: Directory containing images
        batch_size: Batch size for both datasets
    
    Returns:
        train_dataset, val_dataset
    """
    train_dataset = DrivingDataset(
        csv_path=train_csv,
        img_dir=img_dir,
        batch_size=batch_size,
        augment=True,
        is_train=True,
        shuffle=True
    ).create_dataset()
    
    val_dataset = DrivingDataset(
        csv_path=val_csv,
        img_dir=img_dir,
        batch_size=batch_size,
        augment=False,
        is_train=False,
        shuffle=False
    ).create_dataset()
    
    return train_dataset, val_dataset


# Example usage:
# train_dataset, val_dataset = create_dataloaders(
#     train_csv='train.csv',
#     val_csv='val.csv',
#     img_dir='images/',
#     batch_size=32
# )