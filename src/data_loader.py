"""
Production-ready data loader for maize disease classification.

This module provides optimized data loading with automatic train/validation splitting,
performance optimization through caching and prefetching, and automatic label inference
from directory structure.

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - Maize Disease Classification
"""

import tensorflow as tf
import os

# Configuration
DATA_DIR = r'C:\websites\mastersaiproject\data\raw'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


def load_maize_data():
    """
    Loads images from the local drive and splits them into 
    Training (80%) and Validation (20%) sets.
    
    This function implements several production-ready best practices:
    - Automatic 80/20 train/validation split to prevent overfitting
    - Performance optimization via caching and prefetching
    - Automatic label inference from directory structure
    - Consistent random seed for reproducibility
    
    Returns:
        tuple: (train_ds, val_ds, class_names)
            - train_ds: Training dataset (80% of data)
            - val_ds: Validation dataset (20% of data)
            - class_names: List of disease classes found
    
    Example:
        >>> train, val, classes = load_maize_data()
        >>> print(f"Found classes: {classes}")
        Found classes: ['maize_healthy', 'maize_mln', 'maize_streak']
        >>> print(f"Training batches: {len(train)}")
    """
    print(f"Checking directory: {DATA_DIR}")
    
    # 1. Create the training dataset (80% of data)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,  # Fixed seed for reproducibility
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 2. Create the validation dataset (20% of data)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,  # Same seed ensures no overlap between train/val
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Get class names (maize_healthy, maize_streak, maize_mln)
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    print(f"Number of training batches: {len(train_ds)}")
    print(f"Number of validation batches: {len(val_ds)}")

    # 3. Performance optimization (autotuning)
    # This ensures CPU and GPU work in parallel for maximum efficiency
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def get_dataset_info(dataset):
    """
    Extract information about a dataset.
    
    Args:
        dataset: TensorFlow dataset object
    
    Returns:
        dict: Dataset information including batch size, image shape, etc.
    """
    for images, labels in dataset.take(1):
        return {
            'batch_size': images.shape[0],
            'image_shape': images.shape[1:],
            'num_classes': len(tf.unique(labels)[0]),
            'dtype': images.dtype
        }


if __name__ == "__main__":
    # Test the data loader
    print("=" * 60)
    print("MSc AI Project - Data Loader Test")
    print("=" * 60)
    
    try:
        train, val, classes = load_maize_data()
        print("\n✅ Data loader is ready for the Masters Project!")
        print(f"\n📊 Dataset Summary:")
        print(f"   - Classes: {classes}")
        print(f"   - Training batches: {len(train)}")
        print(f"   - Validation batches: {len(val)}")
        
        # Get detailed info
        info = get_dataset_info(train)
        print(f"\n🔍 Batch Details:")
        print(f"   - Batch size: {info['batch_size']}")
        print(f"   - Image shape: {info['image_shape']}")
        print(f"   - Number of classes: {info['num_classes']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have images in the data/raw/ subdirectories:")
        print("   - data/raw/maize_healthy/")
        print("   - data/raw/maize_streak/")
        print("   - data/raw/maize_mln/")
