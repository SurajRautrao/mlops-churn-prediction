# Customer Churn Prediction & Explainability Platform

An end-to-end **MLOps project** that predicts customer churn and provides **actionable business insights** using explainable AI.

---

## Live Demo

*  **Dashboard**: https://mlops-churn-prediction-cfz7bkn6wakz5tsr7mcs76.streamlit.app/
*  **API Docs**: https://churn-api-5fcs.onrender.com/docs

> Both apps run on free tiers and sleep when idle — the first load can take up to a minute.

---

## Problem Statement

Customer churn is a critical challenge for subscription-based businesses.
This project aims to:

* Predict churn probability for individual customers
* Identify key drivers behind churn
* Provide actionable recommendations to reduce churn

---

## Solution Overview

This project combines:

* **Machine Learning** → churn prediction
* **Explainable AI (SHAP)** → transparency
* **Interactive Dashboard** → business usability
* **MLOps practices** → experiment tracking, CI, orchestration and deployment

### Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 19 features (demographics, services, contract and billing), ~26.5% churn rate.

---

## Architecture

```
                 ┌──────────── Training ────────────┐
 data/telco.csv → src/train.py (CatBoost) → MLflow (metrics, models)
                          │
                          ▼
              models/catboost_model.pkl
               │                      │
               ▼                      ▼
   FastAPI /predict (Render)   Streamlit Dashboard (Streamlit Cloud)
               ▲                 • SHAP explanations (local model)
               └── prediction ───• business insights
```

* The **dashboard** sends customer data to the **API** for the churn probability, and loads the same model locally to compute SHAP explanations.
* **Airflow** runs the training pipeline on a schedule; **GitHub Actions** runs training on every push.

---

## Tech Stack

| Layer               | Tools                    |
| ------------------- | ------------------------ |
| ML Model            | CatBoost                 |
| Backend             | FastAPI                  |
| Frontend            | Streamlit                |
| Explainability      | SHAP                     |
| Deployment          | Render + Streamlit Cloud |
| Experiment Tracking | MLflow                   |
| Orchestration       | Airflow                  |
| CI                  | GitHub Actions           |
| Containerization    | Docker                   |

---

## Project Structure

```
├── airflow/dags/churn_pipeline.py   # Daily pipeline: clean data → train
├── api/app.py                       # FastAPI prediction service
├── dashboard/
│   ├── app.py                       # Streamlit dashboard
│   └── explain.py                   # SHAP helper
├── data/telco.csv                   # Raw dataset
├── models/catboost_model.pkl        # Trained model used by API + dashboard
├── src/
│   ├── data.py                      # Load & clean
│   ├── features.py                  # Target / feature split
│   └── train.py                     # Train, evaluate, log to MLflow
├── .github/workflows/ci.yml         # CI: install deps + train
├── Dockerfile                       # API container
└── requirements.txt
```

---

## Features

### Prediction

* Real-time churn probability
* Risk classification: **Low** (< 0.4), **Medium** (0.4 – 0.7), **High** (> 0.7)

### Explainability (SHAP)

* **Waterfall plot** → why this particular customer might churn
* **Beeswarm plot** → what drives churn across customers
* **Dependence plots** → how a feature's value changes its impact
* **Top-5 global feature importance**

### Business Insights

* Key churn drivers per customer
* Rule-based recommendations:

  * Contract upgrade suggestions
  * Pricing optimization
  * Service improvement insights

---

## Dashboard Preview

* Prediction screen:
  <img width="1911" height="816" alt="Prediction tab showing churn probability, risk level and customer metrics" src="https://github.com/user-attachments/assets/31df8054-a63a-4937-ac53-8db36adc58bf" />

* SHAP plots:
  <img width="1360" height="702" alt="SHAP waterfall plot and key insights for a single customer" src="https://github.com/user-attachments/assets/88773b76-a646-4030-b6cd-82569d93ecd9" />
  <img width="1432" height="798" alt="SHAP beeswarm and dependence plots" src="https://github.com/user-attachments/assets/5feee227-41c2-4ce1-a3e5-bfa1a7b00e8c" />
  <img width="1355" height="492" alt="Top global feature importance bar chart" src="https://github.com/user-attachments/assets/3f714ce2-8187-4997-ad5d-d3cb36e32c0c" />

* Business insights:
  <img width="1372" height="800" alt="Business insights tab with churn drivers and recommended actions" src="https://github.com/user-attachments/assets/62a69a73-e83b-4df9-acb6-c32507d8382a" />

---

## Model Details

* Algorithm: **CatBoost Classifier** (handles categorical variables natively)
* Data split: **60 / 20 / 20** train / validation / test, stratified on churn
* Validation set selects the number of boosting iterations; the final model is then refit on train + validation
* Evaluation Metric: **ROC-AUC**
* Performance: **0.845** on the held-out test set

---

## Deployment

* **Backend** — FastAPI, containerized with Docker and hosted on Render
* **Frontend** — Streamlit dashboard hosted on Streamlit Cloud

---

## Installation (Local Setup)

Requires **Python 3.10**.

```bash
git clone https://github.com/SurajRautrao/mlops-churn-prediction.git
cd mlops-churn-prediction

pip install -r requirements.txt
```

Run all commands below from the project root.

### Train the Model

```bash
python -m src.train
mlflow ui          # browse runs at http://localhost:5000
```

This prints the test ROC-AUC, logs the run to MLflow and overwrites `models/catboost_model.pkl`.

### Run API

```bash
uvicorn api.app:app --reload
```

Example request (all fields are optional; missing ones use defaults):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 95, "TotalCharges": 190, "InternetService": "Fiber optic", "Contract": "Month-to-month"}'
```

```json
{"churn_probability": 0.79}
```

Interactive docs: http://localhost:8000/docs

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

> **Note:** the dashboard calls the deployed Render API for predictions (URL set in `dashboard/app.py`). To use your local API, change that URL to `http://localhost:8000/predict`.

---

## Pipeline (Airflow)

The `churn_training_pipeline` DAG runs daily:

1. **load_data** — clean the raw dataset (`src/data.py`)
2. **train_model** — train, evaluate and log the model (`src/train.py`)

The dataset is currently static, so scheduled runs reproduce the same model; the pipeline is ready for a refreshed data source.

---

## Business Impact

This solution helps companies to:

* Target retention efforts at high-risk customers
* Understand which factors drive churn
* Improve retention strategies with data-backed recommendations

---
