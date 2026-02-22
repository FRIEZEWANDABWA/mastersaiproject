# Research Summary: ResNet50 vs CNN for Maize Leaf Disease Classification

**Source:** Nkuna, B.L., Abutaleb, K., Chirima, J.G. et al. (2025).  
*Identification of maize leaf diseases using convolutional neural networks.*  
**Smart Agricultural Technology**, Elsevier. DOI: 10.1016/j.atech.2025.100xxx  
*(Article ID: S2772375525004575)*

---

## Overview

This study evaluated the performance of two CNN architectures — a **standard CNN** and **ResNet50 (transfer learning)** — on the task of classifying maize leaf disease states from field-condition RGB images.

The research is directly relevant to this OUK MSc project because:
1. It uses the same input modality (smartphone RGB images)
2. It evaluates performance in realistic field conditions in Sub-Saharan Africa
3. It quantifies the gap between a baseline CNN and a transfer learning approach
4. It highlights the **notable lack of research on detecting multiple diseases simultaneously at the sub-field scale** — the exact research gap this project addresses

---

## Key Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Standard CNN | 71.01% | Baseline |
| **ResNet50 (Transfer Learning)** | **78.76%** | Best performer |

> **Implication for this project:** ResNet50 with ImageNet pre-training outperforms a trained-from-scratch CNN by **~8 percentage points** under field conditions, a substantial improvement given the small dataset sizes typical of local farm studies.

---

## Architecture Used

- **Input:** RGB images (224 × 224 × 3)
- **Base model:** ResNet50 with ImageNet weights
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Training approach:** Transfer learning with frozen base, fine-tuning of upper layers

These parameters are directly adopted as defaults in `config.py` for this project.

---

## Research Gap Identified

> *"There is a notable lack of research on detecting multiple diseases at the sub-field scale, particularly in smallholder farming contexts in East Africa."*

**This project directly addresses this gap** by:
- Focusing on Uasin Gishu County, Kenya (smallholder farming context)
- Targeting 3 specific disease states (healthy, MSV, MLN) simultaneously
- Adding XAI (Grad-CAM) for farmer-facing explanation — not addressed in Nkuna et al.

---

## Implications for Project Architecture

Based on this study, the following design decisions were made:

1. **Switch from MobileNetV2 to ResNet50** as the primary backbone (`src/model_builder.py`)
2. **Set Learning Rate = 0.001 (Adam)** to match benchmark conditions (`config.py`)
3. **Loss = Categorical Cross-Entropy** as per benchmark (`config.py`)
4. **Input shape = 224 × 224 × 3** (matched to benchmark)
5. **Add Grad-CAM XAI** as an extension beyond Nkuna et al. to address farmer trust

---

## Citation (BibTeX)

```bibtex
@article{nkuna2025maize,
  title   = {Identification of maize leaf diseases using convolutional neural networks},
  author  = {Nkuna, B.L. and Abutaleb, K. and Chirima, J.G. and others},
  journal = {Smart Agricultural Technology},
  year    = {2025},
  publisher = {Elsevier},
  note    = {Article S2772375525004575}
}
```
