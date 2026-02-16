"""
Image preprocessing module for maize disease classification.

This module provides utilities for loading, resizing, and normalizing
images for deep learning model training and inference.
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load, resize, and normalize an image for model input.
    
    Args:
        image_path (str): Path to the input image file
        target_size (Tuple[int, int]): Target dimensions (width, height).
                                       Default is (224, 224) for standard CNNs.
    
    Returns:
        np.ndarray: Preprocessed image array with values normalized to [0, 1]
                   and shape (height, width, channels)
    
    Example:
        >>> img = preprocess_image('data/raw/maize_healthy/sample.jpg')
        >>> print(img.shape)
        (224, 224, 3)
        >>> print(img.min(), img.max())
        0.0 1.0
    """
    # Load image in color mode (BGR by default in OpenCV)
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target dimensions
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def preprocess_batch(
    image_paths: list,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        image_paths (list): List of paths to image files
        target_size (Tuple[int, int]): Target dimensions for all images
    
    Returns:
        np.ndarray: Batch of preprocessed images with shape 
                   (batch_size, height, width, channels)
    """
    batch = [preprocess_image(path, target_size) for path in image_paths]
    return np.array(batch)


if __name__ == "__main__":
    # Example usage
    print("Image preprocessing module loaded successfully.")
    print("Use preprocess_image() to process individual images.")
    print("Use preprocess_batch() to process multiple images at once.")
