"""
Dry-run validation — 1 epoch end-to-end pipeline check.

Verifies every component works without touching the main config:
    data loading → augmentation → ResNet50 build → 1-epoch train → save

Usage: python src/dry_run.py
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF C++ info logs

import tensorflow as tf
import config
from src.data_loader    import load_maize_data
from src.augmentation   import apply_augmentation
from src.model_builder  import build_resnet50_model

SEP = "=" * 60

def dry_run():
    print(SEP)
    print("  MSc AI — Maize Disease Classifier — DRY RUN")
    print("  (1 epoch only — pipeline validation)")
    print(SEP)

    # ── 1. TensorFlow version ────────────────────────────────────────
    print(f"\n✅  TensorFlow : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"    GPU devices : {gpus if gpus else 'None (CPU mode)'}")

    # ── 2. Load data ─────────────────────────────────────────────────
    print("\n📂 Step 1 — Loading data...")
    train_ds, val_ds, class_names = load_maize_data()
    print(f"    Classes     : {class_names}")

    # Count images
    n_train = sum(1 for _ in train_ds.unbatch())
    n_val   = sum(1 for _ in val_ds.unbatch())
    print(f"    Train images: {n_train}")
    print(f"    Val images  : {n_val}")

    if n_train == 0:
        print("\n❌  No training images found in data/raw/. Aborting.")
        return

    # ── 3. Augmentation ──────────────────────────────────────────────
    print("\n🔄 Step 2 — Applying augmentation...")
    train_ds_aug = apply_augmentation(train_ds)
    print("    Augmentation pipeline applied ✅")

    # ── 4. Build ResNet50 model ───────────────────────────────────────
    print("\n🏗️  Step 3 — Building ResNet50 model...")
    model = build_resnet50_model(fine_tune=False)
    total_params     = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w)
                           for w in model.trainable_weights)
    print(f"    Total params     : {total_params:,}")
    print(f"    Trainable params : {trainable_params:,}")
    print(f"    Input shape      : {config.INPUT_SHAPE}")
    print(f"    Optimizer        : Adam  |  LR : {config.LEARNING_RATE}")
    print(f"    Loss             : {config.LOSS}")
    print("    Model built ✅")

    # ── 5. 1-epoch training ───────────────────────────────────────────
    print("\n🚀 Step 4 — Running 1 training epoch...")
    history = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=1,
        verbose=1
    )
    acc     = history.history.get('accuracy',     [None])[0]
    val_acc = history.history.get('val_accuracy', [None])[0]
    loss    = history.history.get('loss',         [None])[0]
    print(f"\n    Train accuracy : {acc:.4f}"    if acc     else "    Train accuracy : N/A")
    print(f"    Val accuracy   : {val_acc:.4f}" if val_acc else "    Val accuracy   : N/A (too few val images)")
    print(f"    Train loss     : {loss:.4f}"    if loss    else "    Train loss     : N/A")

    # ── 6. Save model ─────────────────────────────────────────────────
    print("\n💾 Step 5 — Saving dry-run model...")
    dry_run_path = config.MODELS_DIR / 'dry_run_model.keras'
    model.save(str(dry_run_path))
    print(f"    Saved → {dry_run_path} ✅")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ✅ ALL STEPS PASSED — Full pipeline is working!")
    print(f"  Dataset    : {n_train} train  |  {n_val} val images")
    print(f"  Architecture: ResNet50 (ImageNet) + custom head")
    print(f"  Next step  : Add 46+ images to data/raw/maize_streak/")
    print(f"               and data/raw/maize_mln/  then run:")
    print(f"               python src/train.py")
    print(SEP)


if __name__ == "__main__":
    dry_run()
