"""
Data augmentation pipeline for maize disease classification.

Augmentation artificially expands the training set and improves model
generalisation. Transformations are applied online (on GPU) during
training only — validation and test images are never augmented.

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

import tensorflow as tf
import sys
from pathlib import Path

# Allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def get_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Build and return a Keras Sequential augmentation pipeline.

    The pipeline is applied on-the-fly during training to reduce
    overfitting by creating diverse image variations.

    Transformations applied:
        - Random horizontal & vertical flip
        - Random rotation (±15 degrees)
        - Random zoom
        - Random brightness adjustment
        - Random contrast adjustment

    Returns:
        tf.keras.Sequential: Augmentation model (call on image tensors)

    Example:
        >>> augment = get_augmentation_pipeline()
        >>> augmented_batch = augment(batch_images, training=True)
    """
    layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(config.AUGMENT_ROTATION),
        tf.keras.layers.RandomZoom(config.AUGMENT_ZOOM),
        tf.keras.layers.RandomBrightness(config.AUGMENT_BRIGHTNESS),
        tf.keras.layers.RandomContrast(config.AUGMENT_CONTRAST),
    ]

    return tf.keras.Sequential(layers, name="augmentation")


def apply_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply augmentation to a tf.data.Dataset.

    Args:
        dataset: Training tf.data.Dataset (images, labels)

    Returns:
        tf.data.Dataset: Dataset with augmentation applied inline
    """
    augment = get_augmentation_pipeline()

    def augment_fn(image, label):
        image = augment(image, training=True)
        return image, label

    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":
    pipeline = get_augmentation_pipeline()
    print("✅ Augmentation pipeline built successfully.")
    pipeline.summary()
