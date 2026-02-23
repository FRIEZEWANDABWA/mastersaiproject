"""
scripts/create_test_subsets.py
MaizeGuard AI — Robustness Test Subset Generator

Generates 4 structured test subsets from the held-out test images:
  1. normal          — original unaltered images
  2. low_light       — brightness reduced to 40%
  3. high_brightness — brightness boosted to 180% + contrast 130%
  4. partial_occlusion — random black rectangles (20-40% of image area)

Usage:
  python scripts/create_test_subsets.py --test-dir data/test --out-dir experiments/robustness/subsets

Always run AFTER the train/val/test split has been saved to disk.
"""

import argparse, random, shutil
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

SUBSETS = ['normal', 'low_light', 'high_brightness', 'partial_occlusion']
CLASSES = ['maize_healthy', 'maize_streak', 'maize_mln']
SEED    = 42
random.seed(SEED)
np.random.seed(SEED)


def brighten(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def contrast_boost(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def add_occlusion(img):
    """Add 1–3 random black rectangles covering 20–40% of the image."""
    arr = np.array(img.convert('RGB'))
    h, w = arr.shape[:2]
    total_area = h * w
    covered = 0
    target  = random.uniform(0.20, 0.40) * total_area
    for _ in range(3):
        if covered >= target:
            break
        rh = random.randint(h // 8, h // 3)
        rw = random.randint(w // 8, w // 3)
        y  = random.randint(0, h - rh)
        x  = random.randint(0, w - rw)
        arr[y:y+rh, x:x+rw] = 0
        covered += rh * rw
    return Image.fromarray(arr)


def apply_transform(img, subset_name):
    if subset_name == 'normal':
        return img
    elif subset_name == 'low_light':
        return brighten(img, 0.4)
    elif subset_name == 'high_brightness':
        return contrast_boost(brighten(img, 1.8), 1.3)
    elif subset_name == 'partial_occlusion':
        return add_occlusion(img)
    return img


def process_subset(test_dir: Path, out_dir: Path, subset: str):
    subset_out = out_dir / subset
    count = 0
    for cls in CLASSES:
        cls_in  = test_dir / cls
        cls_out = subset_out / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        if not cls_in.exists():
            print(f"  [WARN] {cls_in} not found — skipping")
            continue
        for img_path in cls_in.glob('*'):
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                out_img = apply_transform(img, subset)
                out_img.save(cls_out / img_path.name)
                count += 1
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
    print(f"  ✅ {subset:20s} — {count} images written to {subset_out}")


def main():
    parser = argparse.ArgumentParser(description='Generate robustness test subsets')
    parser.add_argument('--test-dir', default='data/test',
                        help='Path to held-out test image folder')
    parser.add_argument('--out-dir', default='experiments/robustness/subsets',
                        help='Output directory for all subsets')
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    out_dir  = Path(args.out_dir)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory '{test_dir}' not found. "
            "Run 70/15/15 split first to create data/test/."
        )

    print(f"\n🌽 MaizeGuard AI — Robustness Subset Generator")
    print(f"   Source : {test_dir}")
    print(f"   Output : {out_dir}\n")

    for subset in SUBSETS:
        process_subset(test_dir, out_dir, subset)

    print(f"\nDone. 4 subsets written to {out_dir}/")
    print("Next: python src/evaluate.py --robustness\n")


if __name__ == '__main__':
    main()
