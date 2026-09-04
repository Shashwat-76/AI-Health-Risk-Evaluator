# AI Health Risk Evaluator

An educational machine-learning prototype that uses historical health measurements to estimate a **0–100 continuous risk score**, classify it as **Low / Moderate / High**, explain rule-based contributing factors, and visualize historical trends.

## Features

- Python + pandas data pipeline
- Random Forest **regression** model
- Continuous 0–100 risk score
- Low / Moderate / High risk categorization
- Historical trend analysis by patient
- Streamlit dashboard
- Model persistence with joblib
- Clean project structure suitable for GitHub

## Architecture

```text
Historical CSV
     ↓
Data validation / preprocessing
     ↓
Random Forest Regressor
     ↓
0–100 risk score
     ↓
Low / Moderate / High
     ↓
Explanation + historical trend chart
```

## Project structure

```text
AI-Health-Risk-Evaluator/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── health_data.csv
├── models/
│   └── risk_model.pkl
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── train_model.py
    └── health_agent.py
```

## Run locally

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Train the regression model

```powershell
python src/train_model.py
```

This creates:

```text
models/risk_model.pkl
```

### 4. Start the Streamlit application

```powershell
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How the model works

The model receives:

- Age
- Glucose
- Total cholesterol
- Systolic blood pressure
- BMI

The regression model predicts a continuous risk score. The application maps that score to:

- **0–30:** Low
- **31–60:** Moderate
- **61–100:** High

These thresholds are **prototype thresholds**, not clinical guidelines.

## Historical trend analysis

The CSV contains repeated measurements for each synthetic patient. The dashboard lets you select a patient and plot:

- Glucose
- Cholesterol
- Systolic blood pressure
- BMI
- Risk score

This makes the project demonstrate both **current evaluation** and **longitudinal health-data analysis**.

## Important limitation

The included dataset and risk-score target are **synthetic and created for software demonstration**. The model must not be presented as clinically validated, and it should not be used for diagnosis or treatment decisions.

For a production/clinical system, the target would need to be based on an appropriate, validated outcome, with clinical review, representative data, rigorous validation, calibration, privacy controls, and regulatory review.

## GitHub

After testing locally:

```powershell
git init
git add .
git commit -m "Initial AI health risk evaluator"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

## Interview explanation

> "I built an educational health-risk screening prototype using historical measurements. I used a Random Forest regression model to estimate a continuous 0–100 risk score, then converted that score into Low, Moderate, or High categories. I added a rule-based explanation layer and a Streamlit dashboard for historical trend analysis. The current dataset and target are synthetic, so I would not claim clinical validity."

