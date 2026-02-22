"""
ResNet50 Model Builder — Maize Disease Classification.

Replaces the earlier MobileNetV2 approach based on the 2025 benchmark:
  Nkuna et al. (2025) - Smart Agricultural Technology, Elsevier
  ResNet50 achieved 78.76% accuracy vs 71.01% for standard CNN
  in field-condition RGB image classification.

Author: WANDABWA Frieze (ST62/55175/2025)
Supervisor: Dr. Richard Rimiru (rrimiru@ouk.ac.ke)
Appointed: Feb 18, 2026
Project: MSc AI - OUK
"""

import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def build_resnet50_model(num_classes: int = config.NUM_CLASSES,
                         fine_tune: bool = False) -> tf.keras.Model:
    """
    Build a ResNet50-based transfer learning classifier.

    Architecture choice is grounded in:
      Nkuna et al. (2025) — ResNet50 outperformed standard CNN with
      78.76% vs 71.01% accuracy on field-condition RGB maize leaf images.
      Published in Smart Agricultural Technology, Elsevier.

    Architecture:
        Input (224 × 224 × 3)
          └── ResNet50 (ImageNet weights, frozen or partially unfrozen)
                └── GlobalAveragePooling2D
                      └── Dense 512, ReLU + BatchNorm + Dropout 0.5
                            └── Dense 256, ReLU + Dropout 0.3
                                  └── Dense num_classes, Softmax

    Args:
        num_classes (int): Number of output classes. Default: 3 (config.NUM_CLASSES)
        fine_tune (bool): If True, unfreeze the top FINETUNE_LAYERS for
                          domain-specific feature adaptation.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.

    Example:
        >>> model = build_resnet50_model()
        >>> model.summary()
    """
    # ── 1. Pre-trained ResNet50 base ────────────────────────────────
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*config.IMG_SIZE, config.CHANNELS),
        include_top=False,          # Remove ImageNet classifier head
        weights='imagenet'
    )
    base_model.trainable = False    # Frozen during Phase 1

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-config.FINETUNE_LAYERS]:
            layer.trainable = False

    # ── 2. ResNet50 preprocessing expects inputs in [-1, 1] ─────────
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    # ── 3. Enhanced classifier head (deeper than MobileNetV2 version)
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, config.CHANNELS),
                            name="image_input")
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # First dense block
    x = tf.keras.layers.Dense(512, name="fc_512")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_512")(x)
    x = tf.keras.layers.Activation('relu', name="relu_512")(x)
    x = tf.keras.layers.Dropout(0.5, name="drop_512")(x)

    # Second dense block
    x = tf.keras.layers.Dense(256, activation='relu', name="fc_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_256")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name="predictions")(x)

    model = tf.keras.Model(inputs, outputs,
                           name="MaizeDiseaseClassifier_ResNet50")

    # ── 4. Compile with research-benchmark parameters ─────────────────
    lr = config.LR_FINETUNE if fine_tune else config.LR_FROZEN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS,
        metrics=['accuracy']
    )

    return model


def build_attention_resnet50(num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """
    Build a ResNet50-based model with a Squeeze-Excitation (SE) attention block.

    Inspired by:
      SE-VGG16 MaizeNet (2024-2025) — Squeeze and Excitation attention
      mechanisms improved maize disease classification accuracy.
      Enhanced residual-attention MaizeNet (2025) — F1: 0.9509,
      Accuracy: 0.9595 (NIH/PubMed).

    Adds a channel-wise attention mechanism after the ResNet50 feature
    extraction to help the model focus on the most discriminative leaf
    texture channels (yellowing, necrosis, streak patterns).

    Args:
        num_classes (int): Number of output classes. Default: 3

    Returns:
        tf.keras.Model: Compiled model with SE attention block.
    """
    # ── Base model ────────────────────────────────────────────────────
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*config.IMG_SIZE, config.CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    # ── Squeeze-and-Excitation block ───────────────────────────────────
    def se_block(x, reduction: int = 16):
        """Channel-wise attention: recalibrate feature-map responses."""
        filters = x.shape[-1]
        # Squeeze: global average pooling
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        # Excitation: two-layer FC network
        se = tf.keras.layers.Dense(filters // reduction,
                                   activation='relu',
                                   use_bias=False)(se)
        se = tf.keras.layers.Dense(filters,
                                   activation='sigmoid',
                                   use_bias=False)(se)
        # Scale
        return tf.keras.layers.Multiply()([x, se])

    # ── Full model ────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, config.CHANNELS),
                            name="image_input")
    x = preprocess_input(inputs)
    features = base_model(x, training=False)

    # Apply SE attention to ResNet50 feature maps
    x = se_block(features, reduction=16)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_se")(x)
    x = tf.keras.layers.Dense(512, activation='relu', name="fc_512")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu', name="fc_256")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name="predictions")(x)

    model = tf.keras.Model(inputs, outputs,
                           name="MaizeDiseaseClassifier_SE_ResNet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS,
        metrics=['accuracy']
    )
    return model


def unfreeze_and_fine_tune(model: tf.keras.Model) -> tf.keras.Model:
    """
    Transition model from Phase 1 (frozen base) to Phase 2 (fine-tuning).

    Unfreezes the top FINETUNE_LAYERS of the ResNet50 base and
    recompiles with a lower learning rate to prevent catastrophic forgetting.

    Args:
        model: A compiled Keras model returned by build_resnet50_model().

    Returns:
        tf.keras.Model: Recompiled model ready for fine-tuning.
    """
    base_model = next(l for l in model.layers
                      if isinstance(l, tf.keras.Model))

    base_model.trainable = True
    for layer in base_model.layers[:-config.FINETUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR_FINETUNE),
        loss=config.LOSS,
        metrics=['accuracy']
    )

    trainable = sum(tf.keras.backend.count_params(w)
                    for w in model.trainable_weights)
    print(f"Fine-tuning: {trainable:,} trainable params "
          f"(top {config.FINETUNE_LAYERS} ResNet50 layers unfrozen)")
    return model


if __name__ == "__main__":
    print("Building ResNet50 model (benchmark-aligned)...")
    m = build_resnet50_model()
    m.summary()
    total = m.count_params()
    trainable = sum(tf.keras.backend.count_params(w)
                    for w in m.trainable_weights)
    print(f"\nTotal params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("\nBuilding SE-Attention ResNet50...")
    m_se = build_attention_resnet50()
    m_se.summary()
