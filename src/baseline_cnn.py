"""
src/baseline_cnn.py
MaizeGuard AI — Simple CNN Baseline (Phase 2)

Architecture: 3 × (Conv2D → BatchNorm → MaxPool) → GAP → Dropout → Dense(3)
Purpose: Establish a performance floor for comparison against ResNet50.
Same input size, preprocessing, and training config as the primary model.

Usage:
  python src/baseline_cnn.py
Results saved to: experiments/baseline_cnn/
"""

import sys, os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import (IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR,
                    RANDOM_SEED, DATA_DIR, MODELS_DIR)

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_DIR  = ROOT / 'experiments' / 'baseline_cnn'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['maize_healthy', 'maize_mln', 'maize_streak']
NUM_CLASSES = 3

# ── Build Simple CNN ──────────────────────────────────────────────────────────
def build_simple_cnn(input_shape=(224, 224, 3), num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='SimpleCNN_Baseline')
    return model


# ── Load data ─────────────────────────────────────────────────────────────────
def load_split(subset, shuffle=False):
    data_path = ROOT / 'data' / subset
    if not data_path.exists():
        # Fall back to raw if splits not yet created
        data_path = ROOT / 'data' / 'raw'
        print(f"[WARN] data/{subset} not found — using data/raw (no train/val split)")
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_path),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
        seed=RANDOM_SEED,
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_curves(history, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Simple CNN Baseline — Training Curves', fontweight='bold')
    for ax, metric, title in zip(axs,
                                  ['accuracy', 'loss'],
                                  ['Accuracy', 'Loss']):
        ax.plot(history.history[metric],        label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Val', linestyle='--')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training_curves.png")


def save_confusion_matrix(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Healthy', 'MLN', 'Streak'],
                yticklabels=['Healthy', 'MLN', 'Streak'],
                ax=ax)
    ax.set_title('Simple CNN — Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  🌽 MaizeGuard AI — Simple CNN Baseline Training")
    print("="*55 + "\n")

    # 1. Load datasets
    print("Loading data...")
    train_ds = load_split('train', shuffle=True)
    val_ds   = load_split('val')
    test_ds  = load_split('test')

    # 2. Build model
    model = build_simple_cnn()
    model.summary(print_fn=lambda x: open(OUT_DIR / 'model_summary.txt', 'a').write(x + '\n'))
    print(f"\nModel: {model.name} — {model.count_params():,} parameters\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ROOT / 'models' / 'baseline_cnn_best.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.CSVLogger(str(OUT_DIR / 'training_history.csv')),
    ]

    # 4. Train
    print("Training...\n")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks
    )

    # 5. Evaluate on test set
    print("\n📊 Test Set Evaluation:")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true, y_pred,
        target_names=['Healthy', 'MLN', 'Streak']
    )
    print(report)
    with open(OUT_DIR / 'classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}\n\n")
        f.write(report)

    # 6. Plots
    save_curves(history, OUT_DIR)
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, OUT_DIR)

    print(f"\n✅ All results saved to {OUT_DIR}/")
    print(f"   Test Accuracy: {test_acc:.2%}")
    print("\nNext: python src/train.py  (ResNet50 primary model)")


if __name__ == '__main__':
    main()
