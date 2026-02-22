# Research Summary: MaizeNet Attention Mechanisms

**Source:** Multiple 2025 studies — synthesised below by study type.

### Study A — Enhanced Residual-Attention MaizeNet
*Published:* August 2025 | *Indexed:* NIH/PubMed  
*Title:* "Enhanced residual-attention deep neural network for disease classification in maize leaf images"

### Study B — SE-VGG16 MaizeNet (Squeeze-and-Excitation Attention)
*Published:* 2024–2025 | *Indexed:* IEEE, ResearchGate, Semantic Scholar  
*Title:* "SE-VGG16 MaizeNet: Maize Disease Classification Using Deep Learning and Squeeze and Excitation Attention Networks"

### Study C — MaizeNet with Faster-RCNN + Spatial-Channel Attention
*Published:* 2023–2025 | *Indexed:* ResearchGate  
*Model:* Improved Faster-RCNN + ResNet-50 backbone + spatial-channel attention

---

## Overview

The **MaizeNet** family of models addresses a key limitation of standard CNNs: they treat all spatial regions and feature channels equally, even though disease-relevant features (chlorotic streaks, necrotic spots) occupy only a fraction of the leaf area.

**Attention mechanisms** solve this by learning to focus computational resources on the most discriminative regions — directly relevant to farmer-facing XAI applications.

---

## Key Results

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| ResNet-50 + Spatial-Channel Attention | **97.89%** | — | Multi-class maize disorder, localisation + classification |
| Enhanced Residual-Attention MaizeNet | **95.95%** | **0.9509** | Disease classification, NIH/PubMed |
| SE-VGG16 MaizeNet (SE attention) | ~94% | — | Squeeze-and-excitation channel recalibration |

> **Implication:** Attention mechanisms improve over standard transfer learning (e.g., ResNet50 at 78.76%) by a substantial margin when the dataset is of adequate size.

---

## Attention Mechanisms Explained

### 1. Squeeze-and-Excitation (SE) Attention
Recalibrates **channel-wise** responses by learning which feature maps are most important:
```
Global Avg Pool → FC(reduce) → ReLU → FC(expand) → Sigmoid → Scale feature maps
```
**Why it helps for maize disease:** Streak virus causes yellowing in specific colour channels; SE attention learns to upweight those channels.

### 2. Residual-Attention (MaizeNet)
Combines residual learning (ResNet-style skip connections) with spatial attention:
- **Residual learning**: Improves gradient flow, enabling deeper networks
- **Spatial attention**: Generates a 2D weight map highlighting disease-affected leaf areas

### 3. Spatial-Channel Attention (Faster-RCNN variant)
Combines:
- Spatial attention: *where* to focus (disease lesion location)
- Channel attention: *what* features matter (texture, colour patterns)
- Applied to ResNet-50 backbone for simultaneous localisation + classification

---

## Implications for This Project

| Decision | Rationale |
|----------|-----------|
| `src/model_builder.py` includes `build_attention_resnet50()` | Incorporates SE attention as per MaizeNet research |
| `src/explainability.py` implements Grad-CAM | Provides spatial attention visualisation for farmers |
| Phase 2 plans attention fine-tuning | To approach 95%+ accuracy with larger dataset |

---

## Research Gap This Project Addresses (Beyond MaizeNet)

MaizeNet studies used:
- PlantVillage dataset (lab conditions, controlled background)
- Up to 4 disease classes

**This project extends the work by:**
- Field-condition images (Uasin Gishu County, natural backgrounds)
- Disease states specific to East Africa (MSV and MLN — not in PlantVillage)
- Grad-CAM XAI framed for **farmer-facing extension officer use** (not just accuracy metrics)

---

## Citations (BibTeX)

```bibtex
@article{maizenet_attention_2025,
  title   = {Enhanced residual-attention deep neural network for disease
             classification in maize leaf images},
  year    = {2025},
  journal = {NIH/PubMed indexed},
  note    = {F1-score: 0.9509, Accuracy: 0.9595}
}

@article{se_maizenet_2025,
  title   = {SE-VGG16 MaizeNet: Maize Disease Classification Using Deep
             Learning and Squeeze and Excitation Attention Networks},
  year    = {2024-2025},
  journal = {IEEE},
  note    = {Squeeze-and-Excitation channel attention}
}
```
