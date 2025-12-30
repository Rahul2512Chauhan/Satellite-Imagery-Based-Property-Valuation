# 🛰️ Satellite Imagery–Based Property Valuation

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A multimodal machine learning project exploring how satellite imagery complements traditional housing data for property price prediction**

[Features](#-problem-overview) • [Approach](#-project-approach) • [Models](#-models-implemented) • [Setup](#-how-to-set-up-the-project) • [Findings](#-key-findings)

---

</div>

## 🌟 Overview

This project builds an end-to-end **real estate valuation pipeline** that combines:

<table>
<tr>
<td width="50%" valign="top">

### 📊 Structured Data
- Living area & lot size
- Number of bedrooms/bathrooms
- Construction quality metrics
- Geographic coordinates

</td>
<td width="50%" valign="top">

### 🗺️ Visual Context
- Green cover density
- Water body proximity
- Road connectivity patterns
- Urban layout features

</td>
</tr>
</table>

The goal is not just prediction, but **understanding whether and how visual context adds value** to property valuation.

---

## 🔍 Problem Overview

<div align="center">

```mermaid
graph LR
    A[Traditional Features] -->|Limited Context| B[Valuation Model]
    C[Satellite Imagery] -->|Neighborhood Context| B
    B --> D[Enhanced Predictions]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#e8f5e9
```

</div>

Traditional real estate valuation models rely heavily on structured attributes such as:

- ✅ Living area
- ✅ Number of bedrooms and bathrooms
- ✅ Construction quality
- ✅ Geographic coordinates

However, these features often fail to capture **neighborhood-level context**, such as:

- 🌊 Presence of water bodies
- 🌳 Green spaces vs concrete density
- 🛣️ Road connectivity and urban layout

> **Research Question:** Can satellite imagery improve property valuation when combined with tabular data?

---

## 🧠 Project Approach

We follow a **multimodal regression pipeline**:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                      📍 Property Location                        │
│                    (Latitude / Longitude)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌─────────────────────┐       ┌──────────────────────┐
     │   🛰️ Satellite API   │       │  📋 Tabular Features │
     │  Image Acquisition   │       │   • Sqft             │
     └──────────┬───────────┘       │   • Beds/Baths       │
                │                   │   • Quality          │
                ▼                   │   • Year Built       │
     ┌─────────────────────┐       └──────────┬───────────┘
     │  🧠 CNN (ResNet18)   │                  │
     │  Image Embeddings    │                  │
     │     (512-dim)        │                  │
     └──────────┬───────────┘                  │
                │                              │
                └──────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  🔗 Fusion Module   │
                    │  Multimodal ML      │
                    └──────────┬──────────┘
                               │
                               ▼
                        💰 Price Prediction
```

</div>

---

## 📂 Repository Structure

```
satellite-property-valuation/
│
├── 📁 data/
│   ├── raw/                    # Original train & test datasets
│   ├── processed/              # Cleaned CSVs, aligned subsets
│   └── images/                 # Satellite images (NOT committed)
│
├── 📓 notebooks/
│   ├── 01_preprocessing.ipynb        # 🧹 Data cleaning & EDA
│   ├── 02_tabular_model.ipynb        # 📊 Baseline model
│   ├── 03_image_model.ipynb          # 🖼️ Image-only model
│   ├── 04_fusion_model.ipynb         # 🔗 Multimodal fusion
│   ├── 05_grad_cam.ipynb             # 👁️ Explainability
│   └── 06_evaluation.ipynb           # 📈 Final comparison
│
├── 🐍 src/
│   └── data_fetcher.py         # Satellite image acquisition script
│
├── 📤 outputs/
│   └── predictions.csv         # Final test predictions
│
├── 📋 requirements.txt
├── 📖 README.md
└── 🚫 .gitignore
```

---

## 🧪 Models Implemented

<table>
<tr>
<td width="33%" align="center">

### 1️⃣ Tabular-Only
**Strong Baseline**

![Tabular](https://img.shields.io/badge/Type-Baseline-blue?style=flat-square)

Uses structured housing features only with traditional regression models

✅ Strong performance  
✅ Interpretable  
✅ Fast training  

</td>
<td width="33%" align="center">

### 2️⃣ Image-Only
**Vision Model**

![Image](https://img.shields.io/badge/Type-CNN-orange?style=flat-square)

Satellite images → ResNet18 embeddings

⚠️ Some signal  
⚠️ Noisy predictions  
⚠️ Needs context  

</td>
<td width="33%" align="center">

### 3️⃣ Multimodal
**Fusion Model**

![Fusion](https://img.shields.io/badge/Type-Hybrid-purple?style=flat-square)

Early fusion of tabular + image embeddings

❓ Explores improvements  
❓ Critical analysis  
❓ Honest evaluation  

</td>
</tr>
</table>

---

## 📊 Key Findings

<div align="center">

| Model | RMSE | R² | Performance |
|:------|:----:|:--:|:-----------:|
| **Tabular Only** | ⭐ Best | ⭐ High | 🥇 Winner |
| **Image Only** | ⚠️ Weak | ⚠️ Negative | 🔴 Noisy |
| **Multimodal Fusion** | ⬇️ Lower | ⬇️ Lower | 🟡 Did not improve |

</div>

### 🔑 Key Takeaway

> **Structured tabular features provide the strongest predictive signal for property valuation.**
> 
> Satellite imagery captures meaningful neighborhood-level context (greenery, water, roads), but naïve fusion with high-dimensional image embeddings can introduce noise and does not consistently improve predictive performance.

**This highlights the need for selective or attention-based fusion strategies in real-world multimodal systems.**

---

## 👁️ Explainability with Grad-CAM

To understand *what the CNN looks at*, we apply **Grad-CAM** on satellite images.

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                     🔥 ACTIVATION PATTERNS                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  💎 HIGH-VALUE HOMES          │  🏚️ LOW-VALUE HOMES          ║
║  ─────────────────────────    │  ─────────────────────────   ║
║  ✅ Water bodies              │  ❌ Dense rooftops           ║
║  ✅ Green spaces              │  ❌ Concrete-heavy regions   ║
║  ✅ Open layouts              │  ❌ Industrial textures      ║
║  ✅ Road access               │  ❌ Poor connectivity        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

**Validation:** Satellite imagery captures **semantically meaningful spatial cues**, even when it does not directly improve regression metrics.

---

## 🚀 How to Set Up the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation
```

### 2️⃣ Create Virtual Environment

```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set API Key (Mapbox)

Create a `.env` file in the project root:

```env
MAPBOX_TOKEN=your_mapbox_api_key_here
```

### 🛰️ Download Satellite Images (Optional)

Satellite images are not included due to size and API constraints. To download them:

```bash
python src/data_fetcher.py
```

This will fetch satellite images for a stratified subset of properties.

---

## ▶️ Running the Project

<div align="center">

**Recommended Execution Order**

</div>

```mermaid
graph TD
    A[01_preprocessing.ipynb] -->|Clean Data| B[02_tabular_model.ipynb]
    B -->|Baseline| C[03_image_model.ipynb]
    C -->|Vision Model| D[04_fusion_model.ipynb]
    D -->|Multimodal| E[05_grad_cam.ipynb]
    E -->|Explainability| F[06_evaluation.ipynb]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#e0f2f1
```

| Notebook | Purpose | Output |
|:---------|:--------|:-------|
| `01_preprocessing.ipynb` | 🧹 Data cleaning & EDA | Cleaned datasets |
| `02_tabular_model.ipynb` | 📊 Baseline model | Performance metrics |
| `03_image_model.ipynb` | 🖼️ Image-only model | CNN embeddings |
| `04_fusion_model.ipynb` | 🔗 Multimodal fusion | Combined predictions |
| `05_grad_cam.ipynb` | 👁️ Explainability | Activation maps |
| `06_evaluation.ipynb` | 📈 Final comparison | Model rankings |

---

## 📄 Generating Final Predictions

Final predictions on the test dataset are generated using the **best-performing tabular model**:

```
outputs/predictions.csv
```

**Format:**
```csv
id,predicted_price
1,285000
2,342000
...
```

---

## ⚠️ Notes & Limitations

<table>
<tr>
<td>

⚠️ Satellite imagery is treated as a **complementary signal**, not a replacement

</td>
</tr>
<tr>
<td>

⚠️ Naïve fusion can degrade performance due to noisy high-dimensional features

</td>
</tr>
<tr>
<td>

⚠️ Advanced fusion methods (attention, gating, late fusion) are proposed as future work

</td>
</tr>
</table>

---

## 🔮 Future Improvements

<div align="center">

| Enhancement | Impact |
|:------------|:------:|
| 🎯 Attention-based multimodal fusion | High |
| 🔄 Late fusion of predictions | Medium |
| ⏱️ Temporal satellite imagery | High |
| 📈 Socioeconomic context integration | Medium |
| 🏗️ Architecture search (NAS) | Low |
| 🗺️ Multi-scale spatial features | High |

</div>

---

## 🏁 Final Remarks

<div align="center">

**This project emphasizes engineering discipline, experimental rigor, and honest analysis over chasing marginal metric gains.**

</div>

### It demonstrates:

✨ **End-to-end ML system design**  
✨ **Multimodal data handling**  
✨ **Explainability & interpretability**  
✨ **Critical evaluation of results**  

---

<div align="center">

### 📫 Questions or Feedback?

Open an issue or reach out!

**Made with 🛰️ and 🧠**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/satellite-property-valuation?style=social)](https://github.com/yourusername/satellite-property-valuation)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/satellite-property-valuation?style=social)](https://github.com/yourusername/satellite-property-valuation)

</div>

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

</div>