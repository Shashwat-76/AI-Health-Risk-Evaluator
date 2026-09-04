from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.health_agent import HealthEvaluationAgent
from src.preprocessing import load_data

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "health_data.csv"
MODEL_PATH = ROOT / "models" / "risk_model.pkl"

st.set_page_config(
    page_title="AI Health Risk Evaluator",
    page_icon="❤️",
    layout="wide"
)

@st.cache_data
def get_data():
    return load_data(str(DATA_PATH))

@st.cache_resource
def get_agent():
    return HealthEvaluationAgent(MODEL_PATH)

df = get_data()
agent = get_agent()

st.title("❤️ AI Health Risk Evaluator")
st.caption(
    "Educational prototype: uses historical health measurements to "
    "estimate a continuous risk score and categorize it as Low, Moderate, or High."
)

with st.sidebar:
    st.header("Patient evaluation")
    patient_id = st.selectbox("Historical patient", sorted(df["patient_id"].unique()))

    patient_history = df[df["patient_id"] == patient_id].sort_values("date")
    latest = patient_history.iloc[-1]

    st.subheader("Current / latest values")
    age = st.number_input("Age", 1, 120, int(latest["age"]))
    glucose = st.number_input("Glucose (mg/dL)", 40.0, 500.0, float(latest["glucose"]))
    cholesterol = st.number_input("Total cholesterol (mg/dL)", 80.0, 500.0, float(latest["cholesterol"]))
    systolic_bp = st.number_input("Systolic BP (mmHg)", 70.0, 250.0, float(latest["systolic_bp"]))
    bmi = st.number_input("BMI", 10.0, 60.0, float(latest["bmi"]))
    evaluate = st.button("🔍 Evaluate health", use_container_width=True)

st.subheader("Historical trend")

trend_col = st.selectbox(
    "Choose a health metric",
    ["glucose", "cholesterol", "systolic_bp", "bmi", "risk_score"]
)
st.line_chart(
    patient_history.set_index("date")[trend_col],
    use_container_width=True
)

if len(patient_history) >= 2:
    first = float(patient_history.iloc[0][trend_col])
    last = float(patient_history.iloc[-1][trend_col])
    delta = last - first
    st.write(f"**Trend:** {delta:+.1f} from first to latest recorded value.")

st.divider()

if evaluate:
    patient = {
        "age": age,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "systolic_bp": systolic_bp,
        "bmi": bmi,
    }

    score, level = agent.predict(patient)
    factors = agent.explain(patient)

    c1, c2 = st.columns(2)
    c1.metric("Risk score", f"{score}/100")

    if level == "LOW":
        c2.success("🟢 LOW RISK")
    elif level == "MODERATE":
        c2.warning("🟡 MODERATE RISK")
    else:
        c2.error("🔴 HIGH RISK")

    st.subheader("Contributing factors")
    for factor in factors:
        st.write("• " + factor)

    st.subheader("Suggested next step")
    if level == "LOW":
        st.info("Continue healthy habits and routine monitoring.")
    elif level == "MODERATE":
        st.info("Monitor trends and consider discussing the measurements with a healthcare professional.")
    else:
        st.info("The supplied measurements indicate elevated risk; consider professional medical evaluation.")

st.divider()
st.caption(
    "Important: This project is a software/ML demonstration. "
    "It is not medical advice or a diagnosis. The dataset and target are synthetic for demonstration."
)
