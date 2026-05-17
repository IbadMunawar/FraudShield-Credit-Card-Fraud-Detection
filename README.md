# 🛡️ FraudShield — Credit Card Fraud Detection

> An end-to-end Machine Learning web application for real-time credit card fraud detection, built with Streamlit, scikit-learn, XGBoost, SHAP, and Plotly.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📸 App Overview

| Page | Description |
|---|---|
| 📊 **Overview** | KPI dashboard — total transactions, fraud cases, fraud rate, financial totals |
| 🔍 **EDA & Analysis** | Distribution plots, hourly fraud patterns, correlation heatmap, PCA feature analysis |
| 🤖 **ML Models** | Train RF, XGBoost & LR with SMOTE; compare ROC curves, confusion matrices & feature importance |
| 🎯 **Fraud Predictor** | Real-time prediction with interactive sliders + SHAP waterfall explanation |
| 💼 **Business Impact** | Threshold tuner, precision-recall tradeoff, estimated financial savings calculator |

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraudshield.git
cd fraudshield
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `creditcard.csv` in the project root. The file is **not included** in this repository (150 MB). Download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### 5. (Optional) Configure environment variables
```bash
cp .env.example .env
# Edit .env if your dataset lives elsewhere:
# DATASET_PATH=./path/to/creditcard.csv
```

### 6. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| **Web App** | Streamlit |
| **Data** | Pandas, NumPy |
| **ML Models** | scikit-learn, XGBoost |
| **Imbalance Handling** | imbalanced-learn (SMOTE + RandomUnderSampler) |
| **Explainability** | SHAP |
| **Visualisations** | Plotly (dark theme) |
| **Config** | python-dotenv |

---

## 🧠 ML Pipeline

```
creditcard.csv (284,807 rows)
        │
        ▼
  Train / Test Split (80/20, stratified)
        │
        ▼
  RandomUnderSampler → majority class capped at 50k rows
        │
        ▼
  SMOTE → balance minority (Fraud) class
        │
        ├── Logistic Regression
        ├── Random Forest (150 trees)
        └── XGBoost (150 estimators, hist method)
                │
                ▼
        Best model selected by AUC-ROC
                │
                ▼
        SHAP TreeExplainer / LinearExplainer
```

> **Why undersample first?** The full 284k-row dataset with SMOTE is slow on Streamlit Cloud. Capping the majority class at 50k rows before SMOTE cuts training time to ~25 seconds without meaningfully impacting model quality.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (**do not commit `creditcard.csv`** — it is already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set main file to `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   DATASET_PATH = "./creditcard.csv"
   ```
4. Upload `creditcard.csv` via a shared drive / cloud bucket and adjust `DATASET_PATH` accordingly, or use the Streamlit `st.file_uploader` workaround for datasets under 200 MB.

---

## 📁 Project Structure

```
fraudshield/
├── app.py                  # Main Streamlit app (all 5 pages, ~48 KB)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Excludes creditcard.csv, .env, venv/
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

---

## 📊 Dataset

- **Source**: [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Features**: `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class`
- **Class balance**: 99.83% legitimate / 0.17% fraud

---

## 📄 License

MIT — free to use, modify, and distribute.
