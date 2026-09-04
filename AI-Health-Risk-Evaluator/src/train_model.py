from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import FEATURES, load_data, validate_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "health_data.csv"
MODEL_PATH = ROOT / "models" / "risk_model.pkl"

def main():
    df = load_data(str(DATA_PATH))
    validate_data(df)

    X = df[FEATURES]
    y = df["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=8
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH
    )

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
