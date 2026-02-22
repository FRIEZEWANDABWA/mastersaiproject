"""
CNN model architecture for maize disease classification.

Strategy: Transfer Learning with MobileNetV2
─────────────────────────────────────────────
1. Use MobileNetV2 pre-trained on ImageNet as the feature extractor.
2. Freeze the base during initial training (Phase 1).
3. Unfreeze the top layers for fine-tuning (Phase 2).

Rationale for MobileNetV2:
  - Light-weight: suitable for deployment on edge devices (farm tablets)
  - Strong ImageNet features transfer well to plant disease imagery
  - Good accuracy/parameter trade-off for small-to-medium datasets

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def build_model(num_classes: int = config.NUM_CLASSES,
                fine_tune: bool = False) -> tf.keras.Model:
    """
    Build the MobileNetV2-based classification model.

    Architecture:
        Input (224×224×3)
            └─ MobileNetV2 base (ImageNet weights, frozen or partially unfrozen)
                └─ GlobalAveragePooling2D
                    └─ Dense 256, ReLU + Dropout 0.4
                        └─ Dense num_classes, Softmax

    Args:
        num_classes (int): Number of output classes. Default: 3 (config.NUM_CLASSES)
        fine_tune (bool): If True, unfreeze top FINETUNE_LAYERS for fine-tuning.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.

    Example:
        >>> model = build_model()
        >>> model.summary()
    """
    # ── 1. Pre-trained base ─────────────────────────────────────────
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, config.CHANNELS),
        include_top=False,         # Remove original ImageNet classifier
        weights='imagenet'
    )
    base_model.trainable = False   # Frozen during Phase 1

    if fine_tune:
        # Phase 2: unfreeze last FINETUNE_LAYERS layers
        base_model.trainable = True
        for layer in base_model.layers[:-config.FINETUNE_LAYERS]:
            layer.trainable = False

    # ── 2. MobileNetV2 expects inputs rescaled to [-1, 1] ──────────
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # ── 3. Custom classifier head ────────────────────────────────────
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, config.CHANNELS),
                            name="image_input")
    x = preprocess_input(inputs)            # Rescale to [-1, 1]
    x = base_model(x, training=False)       # Feature extraction
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="fc_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="MaizeDiseaseClassifier")

    # ── 4. Compile ────────────────────────────────────────────────────
    lr = config.LR_FINETUNE if fine_tune else config.LR_FROZEN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def unfreeze_and_fine_tune(model: tf.keras.Model) -> tf.keras.Model:
    """
    Transition the model from Phase 1 (frozen) to Phase 2 (fine-tuning).

    Unfreezes the top FINETUNE_LAYERS of the MobileNetV2 base and
    recompiles with a lower learning rate to avoid catastrophic forgetting.

    Args:
        model: A compiled Keras model returned by build_model().

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.

    Example:
        >>> model = build_model()
        >>> # ... Phase 1 training ...
        >>> model = unfreeze_and_fine_tune(model)
        >>> # ... Phase 2 training ...
    """
    # Find the MobileNetV2 base layer
    base_model = next(l for l in model.layers
                      if isinstance(l, tf.keras.Model))

    base_model.trainable = True
    for layer in base_model.layers[:-config.FINETUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_FINETUNE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable = sum(tf.keras.backend.count_params(w)
                    for w in model.trainable_weights)
    print(f"Fine-tuning: {trainable:,} trainable parameters "
          f"(top {config.FINETUNE_LAYERS} MobileNetV2 layers unfrozen)")
    return model


if __name__ == "__main__":
    print("Building Phase 1 model (frozen base)...")
    m = build_model()
    m.summary()

    trainable = sum(tf.keras.backend.count_params(w)
                    for w in m.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w)
                        for w in m.non_trainable_weights)
    print(f"\nTrainable params   : {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
