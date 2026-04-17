# 🏦 Counterfactual Framework Built on Loan Approval Prediction Models

> An intelligent, explainable AI system that not only predicts loan approval outcomes but tells rejected applicants *exactly* what to change to get approved.

---

## 📌 Overview

Traditional loan approval systems return a binary verdict — approved or rejected — with no guidance. This project builds an **end-to-end decision-support system** that combines machine learning-based loan prediction with **counterfactual explanations**, empowering rejected applicants with actionable, realistic improvement plans.

---

## ✨ Key Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Loan Approval Prediction** | Random Forest classifier (ROC-AUC: 0.763) trained on UCI German Credit Dataset + 3,000 synthetic records |
| 2 | **Counterfactual Explanations** | DiCE framework generates up to 3 feasible improvement plans with financial reasoning |
| 3 | **SHAP Explainability** | Waterfall bar chart showing which features helped/hurt the decision |
| 4 | **What-If Simulator** | Interactive sliders for real-time probability updates (no page reload) |
| 5 | **PDF Report Generation** | Downloadable structured report with verdict, SHAP insights, and action plan |
| 6 | **Risk Category Badge** | Low / Moderate / High risk tier classification with colour coding |
| 7 | **Global Feature Importance Chart** | Top 10 model decision drivers shown on page load |
| 8 | **Progress Tracker** | Checklist that updates approval probability as steps are completed |
| 9 | **Reapplication Readiness Score** | Shows expected probability gain if all action plan steps are followed |
| 10 | **Borderline Applicant Alert** | Detects near-threshold applicants (45–55%) and generates 6 enhanced counterfactuals |

---

## 🏗️ System Architecture

```
User (Browser)
     │
     ▼
HTML/CSS/JS Frontend
     │  (REST API calls)
     ▼
Flask Backend (Python)
├── /predict          → Full prediction + counterfactuals + risk badge
├── /shap             → SHAP values for top-10 features
├── /quick_predict    → Fast probability update (What-If / Progress Tracker)
└── /feature_importance → Pre-computed global feature importances
     │
     ▼
ML Pipeline (Scikit-learn)
├── ColumnTransformer (StandardScaler + OneHotEncoder)
├── Random Forest Classifier (tuned, threshold=0.45)
├── SHAP TreeExplainer
└── DiCE Counterfactual Engine
```

---

## 🧠 ML Models Benchmarked

| Model | Test Accuracy | ROC-AUC |
|-------|--------------|---------|
| **Random Forest (Tuned)** ★ | 0.705 | **0.763** |
| XGBoost (Tuned) | 0.710 | 0.758 |
| AdaBoost | 0.695 | 0.751 |
| Logistic Regression | 0.700 | 0.742 |
| SVM (Tuned) | 0.715 | 0.738 |
| KNN | 0.680 | 0.712 |
| Decision Tree | 0.650 | 0.683 |

Random Forest was selected as the final model for its highest and most consistent ROC-AUC score. A custom decision threshold of **0.45** was used to minimise false negatives.

---

## 📂 Project Structure

```
├── notebook/
│   └── loan_approval_pipeline.ipynb   # Full ML pipeline, training, DiCE, SHAP
├── backend/
│   ├── app.py                         # Flask REST API
│   ├── loan_model.pkl                 # Trained Random Forest pipeline
│   ├── shap_explainer.pkl             # Fitted SHAP TreeExplainer
│   ├── shap_features.pkl              # Feature names for SHAP
│   └── feature_importance.json        # Pre-computed top-10 feature importances
├── frontend/
│   └── index.html                     # Full UI (HTML + CSS + Vanilla JS)
├── Procfile                           # Cloud deployment config (Gunicorn)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Run Locally

```bash
python backend/app.py
```

Then open `http://localhost:5000` in your browser.

### Cloud Deployment (Heroku / Render / AWS)

The app is Procfile-ready for any major PaaS platform:

```
web: gunicorn app:app
```

---

## 🗃️ Dataset

- **UCI Statlog German Credit Dataset** (ID: 144) — 1,000 records, 20 features, binary target
- **Synthetic Augmentation** — 3,000 additional records generated using `SDV GaussianCopulaSynthesizer` to improve model robustness while preserving class distribution

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.x |
| **ML** | Scikit-learn, XGBoost, Random Forest, SVM, KNN, AdaBoost, Logistic Regression |
| **Explainability** | SHAP (TreeExplainer), DiCE-ML |
| **Data Augmentation** | SDV (Synthetic Data Vault) — GaussianCopulaSynthesizer |
| **Backend** | Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **PDF Generation** | jsPDF v2.5.1 (client-side CDN) |
| **Dev Environment** | Google Colab / Jupyter Notebook |

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Full prediction with counterfactuals, SHAP, risk badge |
| `/shap` | POST | Top-10 SHAP feature contributions for a given input |
| `/quick_predict` | POST | Fast probability update for What-If / Progress Tracker |
| `/feature_importance` | GET | Pre-computed global feature importance (top 10) |
| `/health` | GET | API health check |

All endpoints accept and return **JSON**.

---

## 🔬 Counterfactual Logic

Feature mutability is classified into three tiers to ensure realistic suggestions:

| Type | Features | Behaviour |
|------|----------|-----------|
| **Immutable** | age, personal_status_sex, foreign_worker, credit_history, job | Never changed |
| **Semi-mutable** | employment_since, savings_account, housing | Only improved, never worsened |
| **Mutable** | credit_amount, duration_months, installment_rate, purpose, etc. | Modified within financial bounds |

Each counterfactual comes with a human-readable **why**, **how**, **timeline**, and **priority** for every suggested change.

---

## 👥 Team

| Name |
|------|
| Kashish Chelwani | 
| Pruthvieraj Ghule | 
| Palak Goswami | 
| Blessings Phiri | 

---

## 🔮 Future Scope

- Integration with live Credit Score APIs for real-time data
- Adaptation for credit cards, mortgages, and SME loans
- SHAP dependence plots for population-level insights
- Automated SMART adjustment tool in the What-If Simulator
- EU AI Act compliance checker module
- Periodic model retraining pipeline for economic adaptability
- React Native / Flutter mobile application

---

## 📄 License

This project was developed for academic purposes under Symbiosis Institute of Technology, Pune. Please contact the authors before reuse.

---

## 📚 Key References

- Mothilal et al. (2020) — DiCE: Diverse Counterfactual Explanations
- Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
- Wachter et al. (2017) — Counterfactual Explanations Without Opening the Black Box
- Patki et al. (2016) — The Synthetic Data Vault
- UCI German Credit Dataset — https://archive.ics.uci.edu/dataset/144
