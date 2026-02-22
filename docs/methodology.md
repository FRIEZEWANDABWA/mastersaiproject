# Research Methodology

**Project:** Maize Disease Classification using Deep Learning  
**Candidate:** WANDABWA Frieze (ST62/55175/2025)  
**Institution:** Open University of Kenya (OUK) — MSc Artificial Intelligence  

---

## 1. Problem Statement

Maize (*Zea mays*) is the primary food-security crop in Kenya, accounting for a major share of caloric intake across Uasin Gishu county and the broader Rift Valley region. Two diseases — **Maize Streak Virus (MSV)** and **Maize Lethal Necrosis (MLN)** — cause yield losses of up to 90% when left undetected. Early visual diagnosis currently relies on extension officers, who are scarce in rural areas.

This research investigates whether a lightweight Convolutional Neural Network (CNN) can accurately classify maize plants into three health states from smartphone photographs, enabling on-device, real-time diagnosis by farmers.

---

## 2. Research Objectives

1. Collect a labelled image dataset of maize plants across three classes: **healthy**, **streak virus (MSV)**, and **Maize Lethal Necrosis (MLN)**.  
2. Develop and train a transfer-learning-based CNN classifier achieving ≥85% validation accuracy.  
3. Apply **Explainable AI (XAI)** via LIME to interpret model decisions and verify alignment with agronomic disease markers.  
4. Evaluate the model's practical applicability to resource-constrained field conditions.

---

## 3. Dataset

| Attribute | Description |
|-----------|-------------|
| **Location** | Uasin Gishu County, Kenya |
| **Classes** | `maize_healthy`, `maize_streak`, `maize_mln` |
| **Target size** | ≥200 images per class (600+ total) |
| **Capture device** | Smartphone (natural field conditions) |
| **Image format** | JPEG / PNG |
| **Image resolution** | Resized to 224 × 224 px for training |

### Data Collection Protocol

- Photographs taken in natural daylight, multiple angles (leaf close-up, whole plant)
- Disease state confirmed by agronomist before labelling
- Images span different growth stages and lighting conditions
- Informed consent obtained from participating farmers

---

## 4. Model Architecture

### Transfer Learning — MobileNetV2

**Rationale:** MobileNetV2 was selected because:
- Pre-trained on ImageNet: rich low-level features (edges, textures) transfer well to plant images
- Lightweight (~3.4M parameters): suitable for deployment on Android/iOS farmer apps
- Depthwise separable convolutions: significantly faster than standard CNNs of comparable accuracy

### Architecture Overview

```
Input (224 × 224 × 3)
  └── MobileNetV2 Base (ImageNet weights)
        └── GlobalAveragePooling2D
              └── Dense 256, ReLU
                    └── Dropout 0.4
                          └── Dense 3, Softmax  ← [healthy, streak, MLN]
```

### Two-Phase Training Strategy

| Phase | Layers Trained | LR | Purpose |
|-------|---------------|-----|---------|
| Phase 1 — Frozen | Classifier head only | 1e-3 | Fast convergence, stable gradients |
| Phase 2 — Fine-tuning | Head + top 20 MobileNetV2 layers | 1e-5 | Domain-specific feature adaptation |

---

## 5. Data Augmentation

To increase effective dataset size and reduce overfitting, the following augmentations are applied **online during training only**:

| Augmentation | Parameter |
|---|---|
| Random horizontal & vertical flip | — |
| Random rotation | ±15° |
| Random zoom | ±15% |
| Random brightness | ±10% |
| Random contrast | ±10% |

---

## 6. Evaluation Metrics

| Metric | Justification |
|---|---|
| **Accuracy** | Primary performance indicator |
| **Precision** | Minimise false disease alarms |
| **Recall** | Minimise missed disease detections (higher priority) |
| **F1-score** | Harmonic mean — used for imbalanced class comparison |
| **Confusion Matrix** | Reveals which disease pairs are most confused |

---

## 7. Explainability (XAI)

LIME (Local Interpretable Model-agnostic Explanations — Ribeiro et al., 2016) is applied post-training to:

1. Identify image superpixels most influential in each prediction
2. Validate that model focus aligns with agronomic disease markers (leaf streaks, necrotic lesions)
3. Provide interpretable output for farmer-facing applications

LIME explanations are generated per class on randomly selected validation images and saved to `results/lime_*.png`.

---

## 8. Implementation Stack

| Component | Tool |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Image preprocessing | OpenCV, Pillow |
| Data analysis | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Explainability | LIME |
| Experiment tracking | TensorBoard |
| Version control | Git / GitHub |
| Environment | Python 3.10+ |

---

## 9. References

See [`docs/references.bib`](references.bib) for full bibliography.

Key references:
- Adhikari, S. (2020). FRIENDS Framework for research topic selection
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" LIME. KDD 2016
- Sandler, M. et al. (2018). MobileNetV2. CVPR 2018
- DeChant, C. et al. (2017). Automated identification of northern leaf blight-infected maize. *Phytopathology*
