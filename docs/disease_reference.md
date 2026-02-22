# Maize Disease Reference Guide

**Source:** Corson Maize NZ — Pests & Diseases (https://www.corsonmaize.co.nz/pests-and-diseases)  
**Project:** MSc AI — Maize Disease Classification (OUK)  
**Author:** WANDABWA Frieze (ST62/55175/2025)  
**Supervisor:** Dr. Richard Rimiru  
**Last updated:** 2026-02-22

---

## Project Classes & Visual Symptoms

This project classifies maize leaf images into **three classes**. The table below maps visual symptoms to each class and explains what the model should learn to detect.

---

## Class 1: `maize_healthy`

**Visual features:**
- Uniform, vibrant green colour across the full leaf
- Smooth leaf surface with no lesions, spots or discolouration
- Straight leaf veins, no yellowing, no necrosis

**What to look for in images:**
- Clean, dark green leaves at all growth stages
- No streaks, rust pustules, lesions or wilting

---

## Class 2: `maize_streak` — Leaf Disease (Streak / Rust / Blight)

This class captures **leaf-level disease symptoms** including Maize Streak Virus (MSV), Northern Leaf Blight (NLB), Common Rust, and Bacterial Leaf Streak. These are the most common early-detectable diseases in East African smallholder maize.

### 2a. Maize Streak Virus (MSV)
**Source symptom description:**
> Characteristic fine, broken yellow-white streaking running parallel to leaf veins. In severe cases leaves become almost completely yellow-white.

**Visual features:**
- Fine, discontinuous yellow/cream streaks **parallel to veins**
- Patchy chlorosis (loss of green colour) in irregular bands
- Younger leaves show symptoms most sharply
- Leaves may remain green between streaks at early stage

### 2b. Northern Leaf Blight (NLB)
**Source symptom description (Corson):**
> Large spindle-shaped lesions, running in the direction of the veins and causing premature drying of the leaves. In damp weather, dark brown fructifications develop, causing the disease to spread.

**Visual features:**
- Long, cigar/spindle-shaped tan/grey lesions (3–15 cm)
- Lesions elongated **along leaf vein direction**
- Dark brown borders around tan necrotic centre
- Impact: 5–12 days from infection to visible symptoms

### 2c. Common Rust
**Source symptom description (Corson):**
> Raised, rust-coloured pustules appear on both sides of the leaf blade. Chlorosis and leaf death may occur in severe infections. Favours humid conditions and moderate temperatures (15–25°C).

**Visual features:**
- Small, **raised reddish-brown pustules** scattered across leaf surface
- Visible on both sides of the leaf
- May coalesce into larger necrotic patches in severe cases

### 2d. Bacterial Leaf Streak / Goss's Wilt
**Visual features:**
- Water-soaked lesions that become tan/brown with wavy margins
- Bacterial frass (dried tan droplets) visible on lesion surface
- Lesions run between veins

---

## Class 3: `maize_mln` — Maize Lethal Necrosis (MLN)

**Source symptom description:**
> MLN is caused by co-infection of Maize Chlorotic Mottle Virus (MCMV) and Sugarcane Mosaic Virus (SCMV). It causes severe yield loss and plant death across East Africa.

**Visual features:**
- **Chlorotic mottle** (irregular yellow/green mottling) starting from youngest leaves
- Necrosis (browning and death) spreads from leaf tips towards base
- Premature whole-plant death — entire field appears brown/dead
- Dead tassels, cob formation failure
- Distinct from streak: MLN shows **mottled mosaic pattern**, not clean parallel streaks

**Key differentiator from maize_streak:**
| Feature | maize_streak | maize_mln |
|---------|-------------|-----------|
| Pattern | Parallel yellow lines | Irregular mottled mosaic |
| Distribution | Individual leaves | Whole plant, starts from centre |
| Severity | Partial | Can cause total plant death |
| Spread | Leaf-to-leaf | Systemic (whole plant) |

---

## Excluded Diseases (Out of Project Scope)

The following diseases appear in the reference photos but are **outside the three-class scope** of this project. They are excluded from training data:

| Disease | Reason excluded |
|---------|----------------|
| Common smut / Corn smut | Affects the cob/ear, not the leaf — different visual domain |
| Aspergillus ear rot | Ear/kernel disease |
| Diplodia/Fusarium ear rot | Ear/kernel disease |
| Fall Armyworm damage | Pest (insect holes), not foliar disease |
| Stalk rot / root necrosis | Below-ground / stalk lesion |

> **Note:** Future work could expand this classifier to include smut and armyworm damage as additional classes, which would significantly increase the societal impact of the tool.

---

## Data Labelling Guidelines

When adding new images to the training dataset:

1. **Leaf-only images preferred** — close-up of a single leaf showing symptoms clearly
2. **Whole-plant images acceptable** for MLN (where whole-plant collapse is the key symptom)
3. **Avoid:** Images with multiple simultaneous diseases (confuses the model)
4. **Avoid:** Dark/blurry/night images — consistent lighting improves accuracy
5. **File format:** `.jpg`, `.png`, `.webp` all supported by the data loader

---

## References

- Corson Maize NZ — Pests & Diseases: https://www.corsonmaize.co.nz/pests-and-diseases
- Nkuna et al. (2025) — ResNet50 vs CNN for maize disease detection. *Elsevier*
- CIMMYT (2016) — Maize Lethal Necrosis: A guide to field diagnosis
