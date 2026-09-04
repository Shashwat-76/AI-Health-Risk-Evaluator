# AI Health Risk Evaluator 🩺🤖

An AI-powered health risk evaluation prototype that analyzes historical health data and uses a machine learning regression model to generate a continuous health risk score from 0–100.

The system categorizes the predicted score into:

- 🟢 Low Risk
- 🟡 Moderate Risk
- 🔴 High Risk

It also provides contributing health factors and visualizes historical health trends through an interactive Streamlit dashboard.

---

## 🚀 Features

- Historical health data analysis
- Machine Learning regression
- Random Forest Regressor
- Continuous risk score from 0–100
- Low / Moderate / High risk classification
- Health-factor explanation
- Historical trend analysis
- Interactive Streamlit dashboard
- Data preprocessing and validation
- Saved trained ML model
- Model performance evaluation

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Joblib
- Streamlit

---

## 📊 Health Parameters

The model uses the following health parameters:

| Parameter | Description |
|---|---|
| Age | Patient age |
| Glucose | Blood glucose level |
| Cholesterol | Total cholesterol |
| Systolic BP | Systolic blood pressure |
| BMI | Body Mass Index |

---

## 🏗️ System Architecture

```text
Historical Health Data
          ↓
Data Cleaning & Validation
          ↓
Feature Processing
          ↓
Random Forest Regression
          ↓
Risk Score (0–100)
          ↓
Risk Classification
          ↓
Low / Moderate / High
          ↓
AI Evaluation & Explanation
          ↓
Streamlit Dashboard
