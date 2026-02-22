"""
Grad-CAM Explainability Module — Maize Disease Classification.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to
generate visual heatmaps showing which leaf regions drove the model's
disease classification decision.

Purpose (from OUK Research Proposal):
    Building farmer trust through AI explainability is a CORE objective.
    Grad-CAM highlights the precise leaf symptoms (streak patterns,
    necrotic lesions) the model identified — making the AI's reasoning
    transparent to extension officers and farmers.

References:
    Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization. ICCV 2017.

    Nkuna et al. (2025) — Identified the need for explainable AI to
    support extension officer adoption. Smart Agricultural Technology,
    Elsevier.

Author: WANDABWA Frieze (ST62/55175/2025)
Supervisor: Dr. Richard Rimiru (rrimiru@ouk.ac.ke)
Project: MSc AI - OUK
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import config
from src.preprocess import preprocess_image


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Core
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_model(model: tf.keras.Model,
                      last_conv_layer_name: str = None) -> tf.keras.Model:
    """
    Create a Grad-CAM model that outputs both feature maps and predictions.

    Grad-CAM requires access to the gradients of the class score
    with respect to the feature maps of the last convolutional layer.

    Args:
        model: Trained Keras model (ResNet50-based).
        last_conv_layer_name: Name of the last conv layer to visualise.
            Defaults to 'conv5_block3_out' (ResNet50's last conv block).

    Returns:
        tf.keras.Model: Sub-model outputting (conv_outputs, predictions).
    """
    if last_conv_layer_name is None:
        # Default to ResNet50's last convolutional block output
        last_conv_layer_name = 'conv5_block3_out'

    # Try to find the layer — search through submodels if needed
    layer = None
    for l in model.layers:
        if l.name == last_conv_layer_name:
            layer = l
            break
        if hasattr(l, 'layers'):  # e.g. nested ResNet50 base model
            for sub_l in l.layers:
                if sub_l.name == last_conv_layer_name:
                    layer = sub_l
                    break

    if layer is None:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found. "
            "Common ResNet50 last conv layers: "
            "'conv5_block3_out', 'conv5_block3_3_relu'"
        )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output, model.output]
    )
    return grad_model


def compute_gradcam(model: tf.keras.Model,
                    image: np.ndarray,
                    class_index: int = None,
                    last_conv_layer_name: str = None) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a given image.

    Args:
        model: Trained Keras model.
        image: Preprocessed image array (H, W, C) in [0, 1].
        class_index: Target class index. If None, uses the predicted class.
        last_conv_layer_name: Name of the final conv layer.

    Returns:
        np.ndarray: Grad-CAM heatmap array (H, W), normalised to [0, 1].
    """
    grad_model = get_gradcam_model(model, last_conv_layer_name)

    # Expand to batch
    img_batch = np.expand_dims(image * 255.0, axis=0)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_batch, tf.float32)
        conv_outputs, predictions = grad_model(inputs)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        class_channel = predictions[:, class_index]

    # Gradient of class score w.r.t. last conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalise
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index, predictions.numpy()[0]


def overlay_gradcam(original_image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        original_image: Original image in [0, 1] float or [0, 255] uint8.
        heatmap: Grad-CAM heatmap in [0, 1], shape (H, W).
        alpha: Heatmap transparency (0 = invisible, 1 = opaque). Default 0.4.

    Returns:
        np.ndarray: Superimposed image, uint8, (H, W, 3).
    """
    # Ensure original is uint8
    if original_image.max() <= 1.0:
        img = (original_image * 255).astype(np.uint8)
    else:
        img = original_image.astype(np.uint8)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap,
                                  (img.shape[1], img.shape[0]))

    # Apply colourmap (JET: blue=low, red=high importance)
    heatmap_coloured = np.uint8(255 * heatmap_resized)
    heatmap_coloured = cv2.applyColorMap(heatmap_coloured, cv2.COLORMAP_JET)
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    # Blend
    superimposed = cv2.addWeighted(img, 1 - alpha,
                                   heatmap_coloured, alpha, 0)
    return superimposed


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_gradcam(image_path: str,
                      model: tf.keras.Model = None,
                      model_path: str = None,
                      class_names: list = None,
                      save_path: str = None,
                      last_conv_layer: str = None):
    """
    Full Grad-CAM pipeline: load image → compute → visualise → save.

    Args:
        image_path: Path to maize plant image.
        model: Pre-loaded Keras model (optional, loads from model_path if None).
        model_path: Path to .h5 model file (used if model is None).
        class_names: List of class names (defaults to config.CLASSES).
        save_path: Output PNG path (auto-generated if None).
        last_conv_layer: Name of last conv layer (auto-detected if None).

    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float,
            'save_path': str
        }
    """
    if class_names is None:
        class_names = config.CLASSES

    # Load model if not provided
    if model is None:
        model_path = model_path or str(config.BEST_MODEL_PATH)
        model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    image = preprocess_image(image_path, target_size=config.IMG_SIZE)

    # Compute Grad-CAM
    heatmap, pred_idx, probabilities = compute_gradcam(
        model, image, last_conv_layer_name=last_conv_layer)
    superimposed = overlay_gradcam(image, heatmap, alpha=0.4)

    pred_class = class_names[pred_idx]
    confidence = probabilities[pred_idx]

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    heatmap_resized = cv2.resize(heatmap,
                                  (image.shape[1], image.shape[0]))
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(red = most influential)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title(f'Overlay\nPred: {pred_class} ({confidence:.1%})',
                      fontsize=12)
    axes[2].axis('off')

    short = pred_class.replace('maize_', '').upper()
    plt.suptitle(
        f'Grad-CAM Explainability — {short}\n'
        f'(Highlighted regions drove the disease classification)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    # Save
    if save_path is None:
        stem = Path(image_path).stem
        save_path = str(config.RESULTS_DIR / f'gradcam_{stem}_{pred_class}.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Grad-CAM saved → {save_path}")
    print(f"   Prediction : {pred_class} ({confidence:.2%})")

    return {
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'save_path': save_path
    }


def batch_gradcam(image_dir: str,
                  model_path: str = None,
                  n_samples: int = 3):
    """
    Generate Grad-CAM explanations for a random sample of images per class.

    Args:
        image_dir: Root directory containing class subdirectories.
        model_path: Path to .h5 model file.
        n_samples: Number of images to explain per class.
    """
    import random
    model_path = model_path or str(config.BEST_MODEL_PATH)
    model = tf.keras.models.load_model(model_path)
    image_dir = Path(image_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    for cls in config.CLASSES:
        cls_dir = image_dir / cls
        if not cls_dir.exists():
            continue
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in image_exts]
        sample = random.sample(images, min(n_samples, len(images)))
        for img_path in sample:
            try:
                visualise_gradcam(str(img_path), model=model)
            except Exception as e:
                print(f"   ⚠️  Skipped {img_path.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM explanation for a maize image')
    parser.add_argument('--image', required=True,
                        help='Path to maize leaf image')
    parser.add_argument('--model', default=None,
                        help='Path to trained model .h5 (optional)')
    parser.add_argument('--layer', default=None,
                        help='Last conv layer name (auto-detected if omitted)')
    args = parser.parse_args()

    result = visualise_gradcam(
        image_path=args.image,
        model_path=args.model,
        last_conv_layer=args.layer
    )
