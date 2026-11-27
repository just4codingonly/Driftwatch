import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
DRIFT_DATA_PATH = PROJECT_ROOT / "data" / "drift_customer_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "monitoring" / "prediction_drift_results.csv"


def main():
    """
    Compare the model's outputs on reference and drifted data.
    No true labels are needed for this check.
    """

    model = joblib.load(MODEL_PATH)

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    drift_data = pd.read_csv(DRIFT_DATA_PATH)

    X_reference = reference_data.drop(columns=["customer_id", "churn"])
    y_reference = reference_data["churn"]

    # Same held-out split as train_model.py and evaluate_model.py
    _, X_reference_test, _, _ = train_test_split(
        X_reference,
        y_reference,
        test_size=0.2,
        random_state=42,
        stratify=y_reference
    )

    X_drift = drift_data.drop(columns=["customer_id", "churn"])

    # Probability of churn (class 1) for every row
    reference_probabilities = model.predict_proba(X_reference_test)[:, 1]
    drift_probabilities = model.predict_proba(X_drift)[:, 1]

    reference_churn_rate = (reference_probabilities >= 0.5).mean()
    drift_churn_rate = (drift_probabilities >= 0.5).mean()

    statistic, p_value = ks_2samp(
        reference_probabilities,
        drift_probabilities
    )

    results = pd.DataFrame([{
        "reference_rows": len(reference_probabilities),
        "drift_rows": len(drift_probabilities),
        "reference_mean_probability": reference_probabilities.mean(),
        "drift_mean_probability": drift_probabilities.mean(),
        "reference_predicted_churn_rate": reference_churn_rate,
        "drift_predicted_churn_rate": drift_churn_rate,
        "ks_statistic": statistic,
        "p_value": p_value,
        "prediction_drift_detected": p_value < 0.05
    }])

    results.to_csv(RESULTS_PATH, index=False)

    print("=" * 50)
    print("        DRIFTWATCH PREDICTION DRIFT")
    print("=" * 50)
    print(f"Reference predicted churn rate : {reference_churn_rate:.2%}")
    print(f"Drift predicted churn rate     : {drift_churn_rate:.2%}")
    print(f"KS statistic on probabilities  : {statistic:.4f}")
    print(f"P-value                        : {p_value:.6g}")
    print(f"Prediction drift detected      : {p_value < 0.05}")
    print(f"\nResults saved at: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
