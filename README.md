# 🏃 Human Activity Recognition

> Classifying 6 physical activities from smartphone sensor data — comparing Logistic Regression, Random Forest, XGBoost, and Bidirectional LSTM with full Bayesian hyperparameter optimization.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit)](https://your-app.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-lightgrey)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## What this project does

A smartphone worn on the waist generates accelerometer and gyroscope readings 50 times per second. This project uses those readings — pre-processed into 561 statistical features — to predict what a person is doing: walking, going upstairs, going downstairs, sitting, standing, or laying down.

We trained and tuned 4 different ML/DL models to find which approach works best, and why.

**Key finding:** Logistic Regression (F1 = 0.9553) outperformed XGBoost and a Bidirectional LSTM. The 561 handcrafted features are largely linearly separable — when feature engineering is already strong, added model complexity doesn't help and can hurt.

---

## Live Demo

🚀 **[Try it on Streamlit →](https://k63ebhufmfybdpygdypjm6.streamlit.app/)**


Upload a CSV with 561 sensor features and get instant activity predictions with confidence scores.

---

## Results

| Model | Baseline F1-Macro | After HPO |
|---|:---:|:---:|
| **Logistic Regression** | **0.9553 ✅** | 0.9423 |
| Bidirectional LSTM | — | 0.9501 |
| Naive LSTM | 0.9482 | — |
| XGBoost | 0.9429 | 0.9482 |
| Random Forest | 0.9249 | 0.9216 |

> Hyperparameter optimization used **Optuna with TPE sampler** and 3-fold cross-validation. Intensive tuning on Random Forest caused slight overfitting, explaining why its HPO version underperformed the baseline — a useful finding in itself.

---

## Hardest classification pairs

| Confusion | Errors | Why |
|---|:---:|---|
| WALKING_UP ↔ WALKING_DOWN | 88 | Near-identical accelerometer frequency patterns |
| STANDING → LAYING | 21 | Both static; low-movement sensor signal |

---

## Project structure

```
HAR-Activity-Recognition/
├── app.py                                      # Streamlit web app
├── requirements.txt
├── best_model.pkl                              # Saved Logistic Regression (best model)
├── scaler.pkl                                  # Fitted StandardScaler
├── label_encoder.pkl                           # Activity label encoder
├── notebooks/
│   └── Human_Activity_Recognition_Project.ipynb
└── README.md
```

### Notebook sections

| Section | What it covers |
|---|---|
| 1. Preprocessing | Load CSVs, quality checks, label encoding, Z-normalization (fit on train only) |
| 2. EDA | Class distribution, PCA & t-SNE projections, correlation heatmap, feature variance |
| 3. Baseline Models | LR, RF, XGBoost, Naive LSTM — default settings |
| 4. HPO with Optuna | Bayesian tuning (TPE) for all models; LSTM upgraded with sliding window + BiLSTM |
| 5. Results & Error Analysis | Confusion matrices, per-class report, misclassification breakdown |

---

## How to run

### Option A — Streamlit app (no setup needed)

[Use the live demo →](https://k63ebhufmfybdpygdypjm6.streamlit.app/)

Upload your own `test.csv` (561 feature columns) and see predictions instantly.

### Option B — Run locally

```bash
git clone https://github.com/your-username/HAR-Activity-Recognition
cd HAR-Activity-Recognition
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Option C — Reproduce the full notebook

1. Open `notebooks/Human_Activity_Recognition_Project.ipynb` in Google Colab
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) and upload `train.csv` + `test.csv` when prompted
3. Run sections in order — each section loads outputs saved by the previous one

---

## Dataset

**UCI Human Activity Recognition Dataset**
- 10,299 total samples (7,352 train / 2,947 test)
- 561 features derived from accelerometer + gyroscope signals
- 30 subjects — no subject overlap between train and test splits
- 6 activity classes, roughly balanced (max ratio ~1.43)
- Zero missing values, zero duplicate rows, zero zero-variance features

[Download from Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) · [Original UCI page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
optuna
tensorflow
streamlit
joblib
```

Install all: `pip install -r requirements.txt`

---

## Academic context

This project was completed as part of an **Artificial Intelligence specialization** at Al Akhawayn University. It was a team academic project focused on the full ML pipeline: preprocessing, EDA, model selection, hyperparameter optimization, and error analysis.
