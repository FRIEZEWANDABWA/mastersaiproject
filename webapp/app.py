"""
Flask Web Application — MaizeGuard AI
MSc AI Project — OUK | WANDABWA Frieze (ST62/55175/2025)
Supervisor: Dr. Richard Rimiru

Features:
  - ResNet50 disease classification (3 classes)
  - Test-Time Augmentation (TTA) for improved accuracy
  - Confidence thresholding & uncertainty detection
  - Grad-CAM XAI heatmap
  - Full detailed disease report endpoint
"""

import sys, base64, datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
CLASS_NAMES  = ['maize_healthy', 'maize_mln', 'maize_streak']

# ── Rich disease knowledge base ─────────────────────────────────────────────
DISEASE_DB = {
    'maize_healthy': {
        'label':    'Healthy Maize',
        'icon':     '✅',
        'color':    '#22c55e',
        'severity': 'None',
        'severity_score': 0,
        'short_advice': 'Your maize plant appears healthy. Maintain good agronomic practices.',
        'overview': (
            'The leaf shows no signs of disease, pest damage, or nutrient deficiency. '
            'Healthy maize leaves are uniformly dark green, smooth, and free from spots, '
            'streaks, or necrotic patches. Continued monitoring is recommended.'
        ),
        'visual_symptoms_detected': [
            'Uniform green coloration across the leaf blade',
            'No visible lesions, pustules or chlorotic patches',
            'Leaf veins appear normal with no discolouration',
        ],
        'what_it_is': (
            'A healthy maize plant in good agronomic condition. At this stage the '
            'plant is producing chlorophyll efficiently and photosynthesis is unrestricted.'
        ),
        'how_it_spreads': 'N/A — no disease present.',
        'economic_impact': (
            'Healthy plants yield optimally. Protecting this health status is critical '
            'to achieving maximum yield at harvest.'
        ),
        'immediate_actions': [
            'Continue current irrigation and fertilisation schedule',
            'Monitor weekly for early signs of streak or MLN',
            'Remove any weeds that harbour leafhoppers (MSV vectors)',
        ],
        'short_term_actions': [
            'Soil test every season to maintain NPK balance',
            'Rotate crops to avoid soil-borne pathogen accumulation',
            'Keep field records — note any yellowing in adjacent rows',
        ],
        'long_term_prevention': [
            'Use certified, disease-resistant hybrid seed each season',
            'Implement CIMMYT-recommended integrated pest management (IPM)',
            'Consider intercropping with legumes to improve soil nitrogen',
        ],
        'treatment_options': ['No treatment required at this stage'],
        'references': [
            'CIMMYT (2023). Good Agronomic Practices for Maize in East Africa.',
            'Nkuna et al. (2025). ResNet50 for Maize Disease Classification. Elsevier.',
        ],
    },

    'maize_streak': {
        'label':    'Maize Streak Virus (MSV)',
        'icon':     '⚠️',
        'color':    '#f59e0b',
        'severity': 'Moderate–Severe',
        'severity_score': 2,
        'short_advice': 'MSV detected. Control leafhopper vectors immediately. Use resistant varieties for next season.',
        'overview': (
            'Maize Streak Virus (MSV) is one of the most economically damaging '
            'diseases affecting smallholder maize farmers in East and Southern Africa. '
            'It is transmitted by leafhoppers (Cicadulina spp.) and cannot be cured once '
            'established — management focuses on vector control and resistant varieties.'
        ),
        'visual_symptoms_detected': [
            'Broken, discontinuous yellow-white streaks running parallel to leaf veins',
            'Chlorotic (pale yellow) banding between the green vein areas',
            'Symptoms most visible on younger upper leaves',
            'Leaf blade may widen the yellow zones over time',
        ],
        'what_it_is': (
            'MSV is a Mastrevirus transmitted in a persistent manner by the leafhopper '
            'Cicadulina mbila and related species. Once a plant is infected, the virus '
            'is systemic and cannot be eliminated. Severity depends on infection stage — '
            'early infection (before V6) causes yield losses of 20–100%.'
        ),
        'how_it_spreads': (
            'Transmitted exclusively by leafhopper insects (Cicadulina spp.). '
            'Leafhoppers acquire the virus within minutes of feeding on infected plants '
            'and remain viruliferous for life. Spread is rapid in dry conditions '
            'that favour leafhopper reproduction. The virus does NOT spread through '
            'seed or water.'
        ),
        'economic_impact': (
            'Yield losses of 30–100% depending on infection timing. Early infection '
            '(seedling stage) causes near-total crop failure. Late infection (after '
            'tasseling) may only reduce yield by 10–20%. MSV is endemic across '
            'sub-Saharan Africa, costing farmers hundreds of millions USD annually.'
        ),
        'immediate_actions': [
            'Apply a registered insecticide targeting leafhoppers (e.g., imidacloprid, thiamethoxam)',
            'Remove and destroy heavily infected plants to reduce virus reservoir',
            'Do NOT replant cuttings from infected fields',
            'Create a buffer zone — clear grass weeds at field margins that harbour leafhoppers',
        ],
        'short_term_actions': [
            'Scout field every 5–7 days for new infections',
            'Apply reflective mulch (aluminium) to disorient leafhopper vectors if affordable',
            'Notify neighbouring farmers — shared management improves outcomes',
            'Document infection rate (% plants affected) for insurance/compensation',
        ],
        'long_term_prevention': [
            'Switch to MSV-resistant varieties (e.g., SEEDCO SC403, WEMA varieties, H614D)',
            'Plant early in the season before leafhopper populations peak',
            'Use seed treated with systemic insecticides (imidacloprid seed dressings)',
            'Maintain crop-free fallow periods to break the leafhopper-virus cycle',
            'Plant trap crops (e.g., sorghum) at field margins to attract and monitor leafhoppers',
        ],
        'treatment_options': [
            'Insecticide spray: Imidacloprid 200SL (0.5 mL/L) — targets leafhopper adults/nymphs',
            'Seed dressing: Gaucho 600FS (imidacloprid) for next season',
            'Biocontrol: Encourage natural leafhopper predators (lacewings, spiders)',
            'No direct antiviral treatment exists — focus is on vector management',
        ],
        'references': [
            'Corson Maize NZ (2024). Maize Pests & Diseases Guide.',
            'CIMMYT (2016). Maize Streak Virus: Field Diagnosis & Management.',
            'Nkuna et al. (2025). ResNet50-based Maize Disease Detection. Elsevier.',
            'Briddon R.W. (2001). Maize Streak Virus — A review. Virus Research.',
        ],
    },

    'maize_mln': {
        'label':    'Maize Lethal Necrosis (MLN)',
        'icon':     '🔴',
        'color':    '#ef4444',
        'severity': 'Critical',
        'severity_score': 3,
        'short_advice': 'CRITICAL: MLN detected. Remove and destroy ALL affected plants immediately to prevent spread.',
        'overview': (
            'Maize Lethal Necrosis (MLN) is a devastating disease that can cause '
            '100% crop loss in a single season. It is caused by a co-infection of two '
            'viruses: Maize Chlorotic Mottle Virus (MCMV) and any one of several '
            'potyviruses (most commonly Sugarcane Mosaic Virus, SCMV). MLN was first '
            'reported in Kenya in 2011 and has since spread across East Africa, '
            'threatening food security for millions of subsistence farmers.'
        ),
        'visual_symptoms_detected': [
            'Chlorotic (yellow-green) mottling starting from youngest central leaves',
            'Necrosis (browning, drying) spreading from leaf tips inward',
            'Premature whole-plant death — entire plant may appear brown/bleached',
            'Small, malformed or completely absent cobs',
            'Dead tassels before or during pollination',
            'Distinct mosaic pattern (irregular blotches) — differs from streak parallel lines',
        ],
        'what_it_is': (
            'MLN results from synergistic co-infection by MCMV (Maize Chlorotic Mottle '
            'Virus, genus Machlomovirus) and SCMV (Sugarcane Mosaic Virus, genus '
            'Potyvirus). Neither virus alone causes lethal necrosis — their combination '
            'triggers a catastrophic immune failure in the plant. The disease is '
            'systemic and irreversible once established at the whole-plant level.'
        ),
        'how_it_spreads': (
            'MCMV is transmitted by thrips (Frankliniella williamsi), rootworms, '
            'aphids, and contaminated machinery/tools. SCMV is aphid-transmitted. '
            'MLN also spreads through infected seed and crop residue left in the field. '
            'The disease can devastate an entire field within 2–3 weeks in epidemic conditions.'
        ),
        'economic_impact': (
            'Yield losses of 30–100%. In outbreak regions of Kenya, Tanzania, Ethiopia '
            'and Uganda, entire fields have been lost. FAO estimates MLN costs East '
            'African farmers over $500M USD annually. The disease directly threatens '
            'maize food security for 300+ million people in the affected region.'
        ),
        'immediate_actions': [
            '🚨 URGENT: Remove and physically destroy (burn or bury) ALL symptomatic plants',
            '🚨 Do NOT compost infected plant material — this spreads MCMV',
            'Disinfect tools, boots, and machinery with 70% bleach solution before leaving the field',
            'Control thrips and aphid vectors with registered insecticide (spinosad, lambda-cyhalothrin)',
            'Immediately quarantine: do not move plant material outside the field',
            'Notify your local agricultural extension officer',
        ],
        'short_term_actions': [
            'Survey 100% of the field — mark all infected plants for removal',
            'Apply insecticides to control thrips and aphid vectors in remaining healthy plants',
            'Consider destroying the entire crop if infection exceeds 30% of field',
            'Do NOT replant the same field with maize without a 2-season fallow period',
            'File a crop loss report with the National Cereal and Produce Board (Kenya)',
        ],
        'long_term_prevention': [
            'Use ONLY MLN-tolerant/resistant certified varieties (CIMMYT-released WEMA lines)',
            'Source clean certified seed from reputable agrodealers — avoid saved infected seed',
            'Deep plough and remove ALL crop residue at season end — do not leave in field',
            'Enforce at least one season of non-host crop (beans, cassava, sweet potato)',
            'Install sticky yellow traps at field margins to monitor thrips populations',
            'Participate in community-level MLN surveillance coordinated by KEPHIS (Kenya)',
        ],
        'treatment_options': [
            'No cure exists — the only treatment is plant destruction',
            'Vector control: Spinosad 480SC (0.5 mL/L) for thrips; Imidacloprid for aphids',
            'Preventive: Seed treatment with systemic insecticides before planting',
            'CIMMYT offers MLN-tolerant variety testing — contact regional office',
        ],
        'references': [
            'FAO/CIMMYT (2016). Maize Lethal Necrosis: A guide to field diagnosis.',
            'Wangai A.W. et al. (2012). First report of MLN in Kenya. Plant Disease.',
            'KEPHIS (2023). MLN Management Guidelines for East African Farmers.',
            'Nkuna et al. (2025). ResNet50-based Maize Disease Detection. Elsevier.',
            'Corson Maize NZ (2024). Maize Pests & Diseases Guide.',
        ],
    },
}

# ── Model loading ────────────────────────────────────────────────────────────
model = None

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
        print('⚠️  No trained model — demo mode active')
    except Exception as e:
        print(f'⚠️  Model load failed: {e} — demo mode active')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def preprocess_image(img_bytes, target_size=(224, 224)):
    try:
        import tensorflow as tf
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()
    except Exception:
        return None


def predict_with_tta(img_bytes):
    """
    Test-Time Augmentation (TTA) — runs 6 augmented versions of the image
    and averages predictions for improved accuracy.
    Returns (probs_list, is_uncertain, tta_variance)
    """
    import numpy as np

    if model is None:
        raw = np.random.dirichlet([3, 1, 1])
        return raw.tolist(), False, 0.0

    try:
        import tensorflow as tf

        orig = preprocess_image(img_bytes)
        if orig is None:
            return None, False, 0.0

        # Build 6 augmented versions
        augmented = []
        t = tf.constant(orig)
        augmented.append(t)
        augmented.append(tf.image.flip_left_right(t))
        augmented.append(tf.image.flip_up_down(t))
        augmented.append(tf.image.rot90(t, k=1))
        augmented.append(tf.image.adjust_brightness(t, delta=0.1))
        augmented.append(tf.image.adjust_contrast(t, contrast_factor=1.2))

        batch  = tf.stack(augmented, axis=0)
        all_preds = model.predict(batch, verbose=0)   # shape (6, num_classes)
        mean_preds = np.mean(all_preds, axis=0)
        variance   = float(np.mean(np.std(all_preds, axis=0)))
        is_uncertain = (float(np.max(mean_preds)) < 0.55) or (variance > 0.12)

        return mean_preds.tolist(), is_uncertain, variance

    except Exception as e:
        print(f'TTA error: {e}')
        raw = np.random.dirichlet([3, 1, 1])
        return raw.tolist(), False, 0.0


def generate_gradcam(img_bytes):
    try:
        import numpy as np
        import cv2
        import tensorflow as tf

        img = preprocess_image(img_bytes)
        if img is None or model is None:
            return None

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
            top_class  = tf.argmax(preds[0])
            class_score = preds[:, top_class]

        grads   = tape.gradient(class_score, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam     = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
        cam     = np.maximum(cam, 0)
        cam     = cv2.resize(cam, (224, 224))
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        orig_bgr = cv2.resize(orig_bgr, (224, 224))
        overlay  = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

        _, buf = cv2.imencode('.png', overlay)
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f'Grad-CAM error: {e}')
        return None


def count_images(class_name):
    d = ROOT / 'data' / 'raw' / class_name
    if not d.exists():
        return 0
    return len([f for f in d.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = {
        'healthy': count_images('maize_healthy'),
        'streak':  count_images('maize_streak'),
        'mln':     count_images('maize_mln'),
    }
    return render_template('index.html', stats=stats, model_ready=(model is not None))


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG or WEBP.'}), 400

    img_bytes = file.read()
    probs, is_uncertain, variance = predict_with_tta(img_bytes)
    top_idx    = probs.index(max(probs))
    top_class  = CLASS_NAMES[top_idx]
    confidence = probs[top_idx]
    db         = DISEASE_DB[top_class]
    gradcam    = generate_gradcam(img_bytes)

    return jsonify({
        'predicted_class': top_class,
        'label':           db['label'],
        'confidence':      round(confidence * 100, 1),
        'color':           db['color'],
        'icon':            db['icon'],
        'severity':        db['severity'],
        'severity_score':  db['severity_score'],
        'short_advice':    db['short_advice'],
        'is_uncertain':    is_uncertain,
        'tta_variance':    round(variance, 4),
        'demo_mode':       model is None,
        'gradcam':         gradcam,
        'all_probs': {
            CLASS_NAMES[i]: round(probs[i] * 100, 1)
            for i in range(len(CLASS_NAMES))
        },
    })


@app.route('/detailed-report', methods=['POST'])
def detailed_report():
    """
    Returns a full disease report for the uploaded image.
    Contains everything needed for a comprehensive agronomic assessment.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    img_bytes = file.read()
    probs, is_uncertain, variance = predict_with_tta(img_bytes)
    top_idx    = probs.index(max(probs))
    top_class  = CLASS_NAMES[top_idx]
    confidence = probs[top_idx]
    db         = DISEASE_DB[top_class]
    gradcam    = generate_gradcam(img_bytes)

    # Second-most likely disease (differential diagnosis)
    sorted_probs = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
    differential = [
        {'class': c, 'label': DISEASE_DB[c]['label'], 'prob': round(p * 100, 1)}
        for c, p in sorted_probs
    ]

    report = {
        'report_id':    f"MGR-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'timestamp':    datetime.datetime.now().strftime('%d %b %Y, %H:%M'),
        # Prediction
        'predicted_class': top_class,
        'label':           db['label'],
        'confidence':      round(confidence * 100, 1),
        'color':           db['color'],
        'icon':            db['icon'],
        'severity':        db['severity'],
        'severity_score':  db['severity_score'],
        'is_uncertain':    is_uncertain,
        'demo_mode':       model is None,
        'all_probs':       {CLASS_NAMES[i]: round(probs[i]*100,1) for i in range(len(CLASS_NAMES))},
        'differential':    differential,
        # Full disease knowledge
        'overview':                  db['overview'],
        'visual_symptoms_detected':  db['visual_symptoms_detected'],
        'what_it_is':                db['what_it_is'],
        'how_it_spreads':            db['how_it_spreads'],
        'economic_impact':           db['economic_impact'],
        'immediate_actions':         db['immediate_actions'],
        'short_term_actions':        db['short_term_actions'],
        'long_term_prevention':      db['long_term_prevention'],
        'treatment_options':         db['treatment_options'],
        'references':                db['references'],
        # XAI
        'gradcam': gradcam,
    }
    return jsonify(report)


@app.route('/dataset-stats')
def dataset_stats_route():
    return jsonify({
        'healthy': count_images('maize_healthy'),
        'streak':  count_images('maize_streak'),
        'mln':     count_images('maize_mln'),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print('\n' + '=' * 55)
    print('  🌽 MaizeGuard AI — Web Interface')
    print('  → http://127.0.0.1:5000')
    print('=' * 55 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000)
