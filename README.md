# Master of Science in AI — OUK

**Candidate:** WANDABWA Frieze  
**Registration:** ST62/55175/2025  
**Institution:** Open University of Kenya (OUK)  
**Supervisor:** Dr. Richard Rimiru — [rrimiru@ouk.ac.ke](mailto:rrimiru@ouk.ac.ke) *(Appointed: Feb 18, 2026)*  
**Repository:** [github.com/FRIEZEWANDABWA/mastersaiproject](https://github.com/FRIEZEWANDABWA/mastersaiproject.git)

---

## Project Overview

This repository contains the research project for the **MSc in Artificial Intelligence** programme at OUK. The project develops a **deep learning system for automatic classification of maize plant diseases** in Uasin Gishu County, Kenya, supporting food security through accessible AI-powered diagnosis.

### Disease Classes

| Class | Description |
|-------|-------------|
| `maize_healthy` | Healthy maize plant |
| `maize_streak` | Maize Streak Virus (MSV) infection |
| `maize_mln` | Maize Lethal Necrosis (MLN) |

---

## Research Framework (FRIENDS)

This project follows the **FRIENDS framework** (Adhikari, 2020):

| Letter | Criterion | Status |
|--------|-----------|--------|
| **F** | Feasible — local farm data + compute | ✅ |
| **R** | Relevant — Kenyan food security | ✅ |
| **I** | Interesting — AI for agricultural impact | ✅ |
| **E** | Ethical — plant-based, permissions obtained | ✅ |
| **N** | Narrow — 3 specific disease states, Uasin Gishu | ✅ |
| **D** | Discipline — Computer Science / AI (DL + XAI) | ✅ |
| **S** | Supervisor — OUK-standard topic | ✅ |

---

## Model Architecture

> **Research basis:** Nkuna et al. (2025), *Smart Agricultural Technology*, Elsevier — ResNet50 achieved **78.76% accuracy** vs 71.01% for standard CNN in field-condition RGB maize disease classification.

```
Input (224×224×3)  — INPUT_SHAPE per research benchmark
  └── ResNet50 (ImageNet weights — frozen → fine-tuned)
        └── Squeeze-Excitation Attention Block   ← MaizeNet-inspired
              └── GlobalAveragePooling2D
                    └── Dense 512, ReLU + BatchNorm + Dropout 0.5
                          └── Dense 256, ReLU + Dropout 0.3
                                └── Dense 3, Softmax
                                      [healthy | streak | MLN]
```

**Optimizer:** Adam (LR = 0.001) | **Loss:** Categorical Cross-Entropy  
**Explainability:** Grad-CAM heatmaps for farmer-trust (OUK proposal core objective)

**Strategy:** Two-phase transfer learning
1. **Phase 1** — Frozen base, train classifier head (LR = 1e-3)  
2. **Phase 2** — Fine-tune top 20 ResNet50 layers (LR = 1e-5)

---

## Project Structure

```
mastersaiproject/
├── config.py                  ← Central hyperparameters & paths
├── requirements.txt
├── data/
│   └── raw/
│       ├── maize_healthy/     ← Place images here
│       ├── maize_streak/      ← Place images here
│       └── maize_mln/         ← Place images here
├── src/
│   ├── data_loader.py         ← TF dataset pipeline
│   ├── preprocess.py          ← Image preprocessing utilities
│   ├── augmentation.py        ← Online data augmentation
│   ├── model.py               ← MobileNetV2 architecture
│   ├── train.py               ← Two-phase training pipeline
│   ├── evaluate.py            ← Metrics, confusion matrix & LIME XAI
│   ├── predict.py             ← Single-image inference CLI
│   └── project_status.py      ← FRIENDS framework compliance checker
├── notebooks/
│   └── 01_data_exploration.ipynb
├── models/                    ← Saved model weights (git-ignored)
├── results/                   ← Plots & evaluation outputs (git-ignored)
├── logs/                      ← TensorBoard logs (git-ignored)
└── docs/
    ├── FRIENDS_Framework.md
    ├── methodology.md
    └── references.bib
```

---

## Setup

```bash
git clone https://github.com/FRIEZEWANDABWA/mastersaiproject.git
cd mastersaiproject
pip install -r requirements.txt
```

---

## Usage

### 1. Check project readiness
```bash
python src/project_status.py
```

### 2. Add your images
Place photos in the data folders (minimum 50 per class, target 200+):
```
data/raw/maize_healthy/
data/raw/maize_streak/
data/raw/maize_mln/
```

### 3. Train the model
```bash
python src/train.py
```
Training curves are saved to `results/`. Best model is saved to `models/best_model.h5`.

### 4. Evaluate
```bash
python src/evaluate.py
```
Generates classification report, confusion matrix, and LIME XAI explanations.

### 5. Predict a new image
```bash
python src/predict.py --image path/to/maize_photo.jpg
```

### 6. Monitor with TensorBoard
```bash
tensorboard --logdir logs/
```

---

## Documentation

- [Research Methodology](docs/methodology.md)
- [FRIENDS Framework Application](docs/FRIENDS_Framework.md)
- [References](docs/references.bib)

---

## License

Academic research project — All rights reserved.  
© 2025 WANDABWA Frieze, Open University of Kenya
