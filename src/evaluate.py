"""
Model evaluation for maize disease classification.

Computes:
  - Per-class accuracy, precision, recall, F1-score
  - Confusion matrix (saved as PNG)
  - LIME explainability for selected misclassified images

Usage:
    python src/evaluate.py              # evaluates best_model.h5
    python src/evaluate.py --model path/to/model.h5

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import config
from src.data_loader import load_maize_data


# ─────────────────────────────────────────────────────────────────────────────
# Core Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = None) -> tf.keras.Model:
    """
    Load a saved Keras model.

    Args:
        model_path: Path to .h5 file. Defaults to config.BEST_MODEL_PATH.

    Returns:
        tf.keras.Model
    """
    path = model_path or str(config.BEST_MODEL_PATH)
    print(f"   Loading model from: {path}")
    return tf.keras.models.load_model(path)


def evaluate_model(model: tf.keras.Model,
                   val_ds: tf.data.Dataset,
                   class_names: list) -> dict:
    """
    Run full evaluation on the validation dataset.

    Args:
        model: Trained Keras model
        val_ds: Validation tf.data.Dataset
        class_names: List of class name strings

    Returns:
        dict: {
            'true_labels': np.array,
            'pred_labels': np.array,
            'val_loss': float,
            'val_accuracy': float
        }
    """
    print("\n📊 Running evaluation...")
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"   Validation Loss     : {loss:.4f}")
    print(f"   Validation Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Collect predictions
    all_true, all_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        # Labels may be integer (sparse) or one-hot depending on data_loader
        label_vals = labels.numpy()
        if label_vals.ndim > 1:
            label_vals = np.argmax(label_vals, axis=1)  # one-hot → integer
        all_true.extend(label_vals.tolist())
        all_pred.extend(np.argmax(preds, axis=1).tolist())

    return {
        'true_labels': np.array(all_true),
        'pred_labels': np.array(all_pred),
        'val_loss': loss,
        'val_accuracy': accuracy
    }


def print_classification_report(true_labels: np.ndarray,
                                 pred_labels: np.ndarray,
                                 class_names: list):
    """
    Print per-class precision, recall, F1-score using sklearn.

    Args:
        true_labels: Ground truth integer labels
        pred_labels: Predicted integer labels
        class_names: List of class name strings
    """
    from sklearn.metrics import classification_report
    print("\n📋 Classification Report:")
    print("-" * 60)
    print(classification_report(true_labels, pred_labels,
                                target_names=class_names, digits=4))


def plot_confusion_matrix(true_labels: np.ndarray,
                           pred_labels: np.ndarray,
                           class_names: list):
    """
    Generate and save a colour-coded confusion matrix.

    Args:
        true_labels: Ground truth integer labels
        pred_labels: Predicted integer labels
        class_names: List of class name strings
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    short_names = [n.replace('maize_', '') for n in class_names]

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (counts)', 'Confusion Matrix (normalised)'],
        ['d', '.2f']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=short_names, yticklabels=short_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

    plt.tight_layout()
    out_path = config.RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n   📊 Confusion matrix saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LIME Explainability (XAI)
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(model: tf.keras.Model,
                      image: np.ndarray,
                      class_names: list,
                      save_path: str = None):
    """
    Generate a LIME explanation for a single image prediction.

    LIME (Local Interpretable Model-agnostic Explanations) highlights
    which superpixels most influenced the model's decision — satisfying
    the explainability requirement for MSc-level research.

    Args:
        model: Trained Keras model
        image: Preprocessed image numpy array (H, W, C) in [0, 1]
        class_names: List of class name strings
        save_path: Path to save the explanation PNG. Auto-generated if None.

    Returns:
        tuple: (predicted_class, confidence, explanation_image)

    Note:
        Requires `lime` package: pip install lime
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("   ⚠️  LIME not installed. Run: pip install lime")
        return None

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        """Wrapper: normalise and predict a batch of uint8 images."""
        images = images.astype(np.float32) / 255.0
        return model.predict(images, verbose=0)

    # Convert [0,1] float image → uint8 for LIME
    image_uint8 = (image * 255).astype(np.uint8)

    explanation = explainer.explain_instance(
        image_uint8,
        predict_fn,
        top_labels=config.NUM_CLASSES,
        hide_color=0,
        num_samples=config.LIME_NUM_SAMPLES
    )

    # Get prediction
    probs = predict_fn(image_uint8[np.newaxis, ...])[0]
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx]

    # Visualise
    temp, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=True,
        num_features=config.LIME_NUM_FEATURES,
        hide_rest=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title(f'Original — Pred: {pred_class} ({confidence:.2%})')
    axes[0].axis('off')

    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
    axes[1].set_title('LIME Explanation (highlighted regions)')
    axes[1].axis('off')

    plt.suptitle(f'XAI Explanation — {pred_class}', fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = str(config.RESULTS_DIR / f'lime_{pred_class}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   🧠 LIME explanation saved → {save_path}")

    return pred_class, confidence, explanation


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained maize disease classifier')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .h5 model file')
    args = parser.parse_args()

    print("=" * 65)
    print("  MSc AI — Maize Disease Classifier — Evaluation")
    print("=" * 65)

    model = load_model(args.model)
    _, val_ds, class_names = load_maize_data()
    results = evaluate_model(model, val_ds, class_names)
    print_classification_report(
        results['true_labels'], results['pred_labels'], class_names)
    plot_confusion_matrix(
        results['true_labels'], results['pred_labels'], class_names)
    print("\n✅ Evaluation complete. Results saved to results/")


if __name__ == "__main__":
    main()
