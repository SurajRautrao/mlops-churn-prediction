## Customer Churn Prediction & Explainability Platform

An end-to-end **MLOps project** that predicts customer churn and provides **actionable business insights** using explainable AI.

---

## Live Demo

*  **Dashboard**: https://mlops-churn-prediction-cfz7bkn6wakz5tsr7mcs76.streamlit.app/
*  **API Docs**: https://churn-api-5fcs.onrender.com/docs

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
* **MLOps practices** → production deployment

---

## Architecture

```
Streamlit Dashboard (Frontend)
        ↓
FastAPI Backend (Render)
        ↓
CatBoost Model
        ↓
SHAP Explainability Engine
```

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
| Containerization    | Docker                   |

---

## Features

### Prediction

* Real-time churn probability
* Risk classification (Low / Medium / High)

### Explainability

* SHAP Waterfall plot (individual prediction)
* SHAP Beeswarm (global insights)
* SHAP Dependence plots (feature interactions)

### Business Insights

* Key churn drivers per customer
* Rule-based recommendations:

  * Contract upgrade suggestions
  * Pricing optimization
  * Service improvement insights

---

## Dashboard Preview

* Prediction screen:
  <img width="1911" height="816" alt="image" src="https://github.com/user-attachments/assets/31df8054-a63a-4937-ac53-8db36adc58bf" />

* SHAP plots:
  <img width="1360" height="702" alt="image" src="https://github.com/user-attachments/assets/88773b76-a646-4030-b6cd-82569d93ecd9" />
  <img width="1432" height="798" alt="image" src="https://github.com/user-attachments/assets/5feee227-41c2-4ce1-a3e5-bfa1a7b00e8c" />
  <img width="1355" height="492" alt="image" src="https://github.com/user-attachments/assets/3f714ce2-8187-4997-ad5d-d3cb36e32c0c" />

* Business insights:
  <img width="1372" height="800" alt="image" src="https://github.com/user-attachments/assets/62a69a73-e83b-4df9-acb6-c32507d8382a" />

---

## Model Details

* Algorithm: **CatBoost Classifier**
* Handles categorical variables natively
* Evaluation Metric: **ROC-AUC**
* Performance: **0.8647**

---

## Explainability (SHAP)

* Local explanations → Why this customer will churn
* Global insights → What drives churn overall
* Feature interactions → How variables influence each other

---

## Deployment

### Backend (FastAPI)

* Hosted on Render
* Public API endpoint available

### Frontend (Streamlit)

* Deployed on Streamlit Cloud
* Fully interactive dashboard

---

## Installation (Local Setup)

```bash
git clone https://github.com/SurajRautrao/mlops-churn-prediction.git
cd mlops-churn-prediction

pip install -r requirements.txt
```

### Run API

```bash
uvicorn api.app:app --reload
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Pipeline (Airflow)

* Data ingestion
* Feature engineering
* Model training
* Scheduled retraining

---

## Business Impact

This solution enables companies to:

* Reduce churn through targeted interventions
* Understand customer behavior
* Improve retention strategies
* Increase revenue

---


