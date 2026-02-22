"""
Central configuration for Maize Disease Classification project.

All hyperparameters and paths are defined here so that any changes
propagate consistently across all modules.

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - OUK
"""

from pathlib import Path

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────
PROJECT_ROOT = Path(r'C:\websites\mastersaiproject')
DATA_DIR     = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR   = PROJECT_ROOT / 'models'
RESULTS_DIR  = PROJECT_ROOT / 'results'
LOGS_DIR     = PROJECT_ROOT / 'logs'

# Ensure runtime directories exist
for _dir in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
CLASSES = ['maize_healthy', 'maize_streak', 'maize_mln']
NUM_CLASSES = len(CLASSES)

# ─────────────────────────────────────────
# Image Settings
# ─────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS   = 3

# ─────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────
BATCH_SIZE       = 32
SEED             = 42

# Phase 1: frozen base (transfer learning)
EPOCHS_FROZEN    = 10
LR_FROZEN        = 1e-3

# Phase 2: fine-tuning (top layers unfrozen)
EPOCHS_FINETUNE  = 20
LR_FINETUNE      = 1e-5
FINETUNE_LAYERS  = 20          # number of top MobileNetV2 layers to unfreeze

# ─────────────────────────────────────────
# Early Stopping & LR Scheduler
# ─────────────────────────────────────────
PATIENCE_EARLY_STOP = 5
PATIENCE_LR_REDUCE  = 3
LR_REDUCE_FACTOR    = 0.5
LR_REDUCE_MIN       = 1e-7

# ─────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────
BEST_MODEL_PATH  = MODELS_DIR / 'best_model.h5'
FINAL_MODEL_PATH = MODELS_DIR / 'final_model.h5'

# ─────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────
AUGMENT_FLIP       = True
AUGMENT_ROTATION   = 0.15      # fraction of 2π
AUGMENT_ZOOM       = 0.15
AUGMENT_BRIGHTNESS = 0.1
AUGMENT_CONTRAST   = 0.1

# ─────────────────────────────────────────
# Evaluation & XAI
# ─────────────────────────────────────────
LIME_NUM_SAMPLES   = 1000      # LIME perturbation samples
LIME_NUM_FEATURES  = 10        # superpixels to highlight
