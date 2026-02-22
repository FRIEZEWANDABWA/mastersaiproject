"""
Training pipeline for maize disease classification.

Implements a two-phase training strategy:
  Phase 1 — Frozen Base  : Train only the classifier head (fast convergence)
  Phase 2 — Fine-tuning  : Unfreeze top MobileNetV2 layers (higher accuracy)

Usage:
    python src/train.py

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import config
from src.data_loader import load_maize_data
from src.augmentation import apply_augmentation
from src.model import build_model, unfreeze_and_fine_tune


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(phase: str) -> list:
    """
    Return a list of Keras callbacks for the given training phase.

    Args:
        phase: 'frozen' or 'finetune'

    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard]
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.PATIENCE_LR_REDUCE,
            min_lr=config.LR_REDUCE_MIN,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.LOGS_DIR / phase),
            histogram_freq=1
        ),
    ]
    return callbacks


def plot_history(history, phase: str):
    """
    Plot and save training accuracy and loss curves.

    Args:
        history: Keras History object from model.fit()
        phase: Label for the plot filename ('frozen' or 'finetune')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title(f'Accuracy — {phase}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'],     label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'Loss — {phase}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = config.RESULTS_DIR / f'training_curves_{phase}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   📊 Saved training curves → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    """
    Execute the full two-phase training pipeline.
    """
    print("=" * 65)
    print("  MSc AI — Maize Disease Classifier — Training")
    print("=" * 65)

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n📂 Loading data...")
    train_ds, val_ds, class_names = load_maize_data()
    print(f"   Classes : {class_names}")

    # ── 2. Apply augmentation (training set only) ───────────────────
    print("\n🔄 Applying data augmentation to training set...")
    train_ds = apply_augmentation(train_ds)

    # ── 3. Phase 1: Frozen base training ────────────────────────────
    print("\n🔒 PHASE 1 — Frozen Base Training")
    print("-" * 65)
    model = build_model(fine_tune=False)
    model.summary(print_fn=lambda x: None)   # suppress verbose summary
    print(f"   Epochs     : {config.EPOCHS_FROZEN}")
    print(f"   LR         : {config.LR_FROZEN}")
    print(f"   Batch size : {config.BATCH_SIZE}")

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=get_callbacks('frozen'),
        verbose=1
    )
    plot_history(history_frozen, 'frozen')

    # ── 4. Phase 2: Fine-tuning ─────────────────────────────────────
    print("\n🔓 PHASE 2 — Fine-Tuning (top layers unfrozen)")
    print("-" * 65)
    model = unfreeze_and_fine_tune(model)
    print(f"   Epochs     : {config.EPOCHS_FINETUNE}")
    print(f"   LR         : {config.LR_FINETUNE}")

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=get_callbacks('finetune'),
        verbose=1
    )
    plot_history(history_finetune, 'finetune')

    # ── 5. Save final model ─────────────────────────────────────────
    model.save(str(config.FINAL_MODEL_PATH))
    print(f"\n✅ Final model saved → {config.FINAL_MODEL_PATH}")
    print("✅ Best model saved  → {config.BEST_MODEL_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    train()
