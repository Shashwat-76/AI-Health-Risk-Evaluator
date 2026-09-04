from pathlib import Path
import joblib
import pandas as pd

class HealthEvaluationAgent:
    """Educational risk-screening agent; not a diagnostic system."""

    def __init__(self, model_path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]

    def predict(self, health_data: dict):
        row = pd.DataFrame([health_data])[self.features]
        score = float(self.model.predict(row)[0])
        score = max(0.0, min(100.0, score))

        if score <= 30:
            level = "LOW"
        elif score <= 60:
            level = "MODERATE"
        else:
            level = "HIGH"

        return round(score, 1), level

    def explain(self, health_data: dict):
        factors = []
        if health_data["glucose"] >= 126:
            factors.append("Glucose is elevated.")
        if health_data["cholesterol"] >= 200:
            factors.append("Total cholesterol is elevated.")
        if health_data["systolic_bp"] >= 140:
            factors.append("Systolic blood pressure is elevated.")
        if health_data["bmi"] >= 25:
            factors.append("BMI is above the normal range.")
        if health_data["age"] >= 60:
            factors.append("Age is contributing to the overall score.")
        return factors or ["No major rule-based flags were triggered."]
