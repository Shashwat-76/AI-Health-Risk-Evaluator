import pandas as pd

FEATURES = ["age", "glucose", "cholesterol", "systolic_bp", "bmi"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["patient_id", "date"]).reset_index(drop=True)
    return df

def validate_data(df: pd.DataFrame) -> None:
    required = {"patient_id", "date", "risk_score", *FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[FEATURES + ["risk_score"]].isna().any().any():
        raise ValueError("Missing numeric values detected.")
