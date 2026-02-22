"""
Single-image inference script for maize disease classification.

Given any photo of a maize plant, this script loads the trained model
and returns the predicted disease class with confidence score.

Usage:
    python src/predict.py --image path/to/maize_photo.jpg
    python src/predict.py --image photo.jpg --model models/best_model.h5

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
import config
from src.preprocess import preprocess_image


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image_path: str,
                  model_path: str = None) -> dict:
    """
    Load the trained model and predict the disease class of a single image.

    Args:
        image_path (str): Path to the input maize plant image.
        model_path (str): Path to the .h5 model file.
                          Defaults to config.BEST_MODEL_PATH.

    Returns:
        dict: {
            'predicted_class': str,   e.g. 'maize_streak'
            'confidence':      float, e.g. 0.9432
            'all_probabilities': dict, {class_name: probability, ...}
        }

    Example:
        >>> result = predict_image('data/raw/maize_healthy/img001.jpg')
        >>> print(result['predicted_class'])
        maize_healthy
        >>> print(f"{result['confidence']:.2%}")
        96.07%
    """
    # ── 1. Load model ───────────────────────────────────────────────
    model_path = model_path or str(config.BEST_MODEL_PATH)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train the model first with: python src/train.py"
        )
    model = tf.keras.models.load_model(model_path)

    # ── 2. Preprocess image ─────────────────────────────────────────
    image = preprocess_image(image_path, target_size=config.IMG_SIZE)

    # Model expects a batch dimension
    image_batch = np.expand_dims(image, axis=0)   # (1, 224, 224, 3)

    # The model's preprocess_input layer expects uint8-like [0,255] values
    # Our preprocess_image returns [0,1]  → rescale back to [0,255]
    image_batch = image_batch * 255.0

    # ── 3. Predict ───────────────────────────────────────────────────
    probabilities = model.predict(image_batch, verbose=0)[0]
    pred_index = int(np.argmax(probabilities))
    pred_class = config.CLASSES[pred_index]
    confidence = float(probabilities[pred_index])

    # Build full probability dict
    all_probs = {cls: float(p)
                 for cls, p in zip(config.CLASSES, probabilities)}

    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }


def print_prediction(result: dict, image_path: str):
    """
    Pretty-print the prediction result to the console.

    Args:
        result: Dictionary returned by predict_image()
        image_path: Path to the image (for display)
    """
    bar_len = 30

    print("\n" + "=" * 55)
    print("  MAIZE DISEASE PREDICTION")
    print("=" * 55)
    print(f"  Image  : {Path(image_path).name}")
    print(f"  Result : {result['predicted_class'].upper()}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print("\n  Class Probabilities:")
    print("  " + "-" * 45)

    for cls, prob in sorted(result['all_probabilities'].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * bar_len)
        short = cls.replace('maize_', '').capitalize()
        print(f"  {short:10s}  {bar:<{bar_len}}  {prob:.2%}")

    print("=" * 55)

    # Plain-language interpretation
    cls = result['predicted_class']
    if cls == 'maize_healthy':
        print("  🌽 Plant appears HEALTHY. No disease detected.")
    elif cls == 'maize_streak':
        print("  ⚠️  Maize Streak Virus (MSV) detected.")
        print("     Recommendation: Isolate affected plants, consult an agronomist.")
    elif cls == 'maize_mln':
        print("  🚨 Maize Lethal Necrosis (MLN) detected.")
        print("     Recommendation: Remove affected plants immediately.")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict maize disease from a plant photo')
    parser.add_argument('--image', required=True,
                        help='Path to the maize plant image')
    parser.add_argument('--model', default=None,
                        help='Path to trained model .h5 file (optional)')
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print_prediction(result, args.image)
