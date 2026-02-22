"""
Flask Web Application — Maize Disease Classifier
MSc AI Project — OUK | WANDABWA Frieze (ST62/55175/2025)
Supervisor: Dr. Richard Rimiru

Serves a web interface for uploading maize leaf images
and returning disease classification + Grad-CAM explanation.
"""

import sys, os, io, base64, json, random
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Flask init ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max upload
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}

# ── Load model (optional — falls back gracefully if not trained yet) ──────────
model = None
CLASS_NAMES = ['maize_healthy', 'maize_mln', 'maize_streak']
CLASS_LABELS = {
    'maize_healthy': 'Healthy',
    'maize_mln':     'Maize Lethal Necrosis (MLN)',
    'maize_streak':  'Maize Streak Virus'
}
CLASS_COLORS = {
    'maize_healthy': '#22c55e',
    'maize_mln':     '#ef4444',
    'maize_streak':  '#f59e0b'
}
CLASS_ICONS = {
    'maize_healthy': '✅',
    'maize_mln':     '🔴',
    'maize_streak':  '⚠️'
}
CLASS_ADVICE = {
    'maize_healthy': 'Your maize plant appears healthy. Continue regular monitoring and good agronomic practices.',
    'maize_mln':     'MLN detected. This is a serious viral disease. Remove and destroy affected plants immediately to prevent spread. Replant with MLN-resistant varieties.',
    'maize_streak':  'Maize Streak Virus detected. Control leafhopper vectors. Consider resistant hybrid varieties.'
}

def load_model():
    global model
    try:
        import tensorflow as tf
        model_paths = [
            ROOT / 'models' / 'best_model.h5',
            ROOT / 'models' / 'final_model.h5',
            ROOT / 'models' / 'dry_run_model.keras',
        ]
        for path in model_paths:
            if path.exists():
                model = tf.keras.models.load_model(str(path))
                print(f'✅ Model loaded: {path.name}')
                return
        print('⚠️  No trained model found — using demo mode')
    except Exception as e:
        print(f'⚠️  Could not load model: {e} — using demo mode')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(img_bytes):
    """Resize and normalise image bytes for model inference."""
    import numpy as np
    try:
        import tensorflow as tf
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()
    except Exception:
        return None

def predict(img_bytes):
    """Run prediction and return class probabilities."""
    if model is not None:
        import numpy as np
        img = preprocess_image(img_bytes)
        if img is not None:
            preds = model.predict(img[None, ...], verbose=0)[0]
            return preds.tolist()

    # Demo mode: plausible-looking random predictions
    import numpy as np
    raw = np.random.dirichlet([3, 1, 1])   # bias toward healthy for demo
    return raw.tolist()

def generate_gradcam(img_bytes):
    """
    Generate Grad-CAM heatmap overlay.
    Returns base64-encoded PNG or None if model unavailable.
    """
    try:
        import numpy as np
        import cv2
        import tensorflow as tf

        img = preprocess_image(img_bytes)
        if img is None or model is None:
            return None

        # Build Grad-CAM model from last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(img[None, ...], tf.float32)
            conv_out, preds = grad_model(inputs)
            top_class = tf.argmax(preds[0])
            class_score = preds[:, top_class]

        grads = tape.gradient(class_score, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        orig_bgr = cv2.resize(orig_bgr, (224, 224))
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

        _, buf = cv2.imencode('.png', overlay)
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f'Grad-CAM error: {e}')
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    dataset_stats = {
        'healthy': count_images('maize_healthy'),
        'streak':  count_images('maize_streak'),
        'mln':     count_images('maize_mln'),
    }
    model_ready = model is not None
    return render_template('index.html',
                           stats=dataset_stats,
                           model_ready=model_ready)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG or WEBP.'}), 400

    img_bytes = file.read()
    probs = predict(img_bytes)
    top_idx = probs.index(max(probs))
    top_class = CLASS_NAMES[top_idx]
    confidence = probs[top_idx]

    # Grad-CAM (optional)
    gradcam_b64 = generate_gradcam(img_bytes)

    result = {
        'predicted_class': top_class,
        'label':      CLASS_LABELS[top_class],
        'confidence': round(confidence * 100, 1),
        'color':      CLASS_COLORS[top_class],
        'icon':       CLASS_ICONS[top_class],
        'advice':     CLASS_ADVICE[top_class],
        'all_probs':  {
            CLASS_NAMES[i]: round(probs[i] * 100, 1)
            for i in range(len(CLASS_NAMES))
        },
        'gradcam':    gradcam_b64,
        'demo_mode':  model is None,
    }
    return jsonify(result)

@app.route('/dataset-stats')
def dataset_stats():
    return jsonify({
        'healthy': count_images('maize_healthy'),
        'streak':  count_images('maize_streak'),
        'mln':     count_images('maize_mln'),
    })

def count_images(class_name):
    d = ROOT / 'data' / 'raw' / class_name
    if not d.exists():
        return 0
    return len([f for f in d.iterdir()
                if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print('\n' + '='*55)
    print('  🌽 Maize Disease Classifier — Web Interface')
    print('  Open in browser: http://127.0.0.1:5000')
    print('='*55 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000)
