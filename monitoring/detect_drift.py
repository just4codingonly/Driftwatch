import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
CURRENT_DATA_PATH = PROJECT_ROOT / "data" / "drift_customer_data.csv"

NUMERICAL_COLUMNS = ["age", "monthly_income", "monthly_usage_hours", "support_calls"]
CATEGORICAL_COLUMNS = ["contract_type"]


def detect_numerical_drift(reference_data, current_data, column):
    statistic, p_value = ks_2samp(reference_data[column], current_data[column])
    return statistic, p_value, p_value < 0.05


def detect_categorical_drift(reference_data, current_data, column):
    reference_counts = reference_data[column].value_counts()
    current_counts = current_data[column].value_counts()
    comparison = pd.DataFrame({
        "reference": reference_counts,
        "current": current_counts
    }).fillna(0)
    statistic, p_value, _, _ = chi2_contingency(comparison)
    return statistic, p_value, p_value < 0.05


def run_experiment(reference_data, current_data, level):
    """Apply controlled changes and calculate the percentage of drifted features."""
    data = current_data.copy()

    factors = {"Low": 0.15, "Medium": 0.40, "High": 0.80}
    factor = factors[level]

    for column in NUMERICAL_COLUMNS:
        reference_mean = reference_data[column].mean()
        data[column] = data[column] + (reference_mean * factor)

    if level == "Medium":
        data["contract_type"] = np.where(
            np.arange(len(data)) % 2 == 0, "Monthly", data["contract_type"]
        )
    elif level == "High":
        data["contract_type"] = "Monthly"

    results = []
    for column in NUMERICAL_COLUMNS:
        statistic, p_value, detected = detect_numerical_drift(
            reference_data, data, column
        )
        results.append(detected)

    for column in CATEGORICAL_COLUMNS:
        statistic, p_value, detected = detect_categorical_drift(
            reference_data, data, column
        )
        results.append(detected)

    drifted = sum(results)
    total = len(results)
    score = (drifted / total) * 100 if total else 0

    return {
        "drift_level": level,
        "total_features": total,
        "drifted_features": drifted,
        "drift_score_percent": round(score, 2),
        "status": "HIGH DRIFT" if score >= 70 else
                  "MEDIUM DRIFT" if score >= 30 else "LOW DRIFT"
    }


def main():
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    experiment_results = [
        run_experiment(reference_data, current_data, level)
        for level in ["Low", "Medium", "High"]
    ]

    output_path = PROJECT_ROOT / "monitoring" / "drift_experiments.csv"
    pd.DataFrame(experiment_results).to_csv(output_path, index=False)

    # Preserve the original detailed result file for the dashboard.
    detailed_results = []
    for column in NUMERICAL_COLUMNS:
        statistic, p_value, detected = detect_numerical_drift(
            reference_data, current_data, column
        )
        detailed_results.append({
            "feature": column,
            "test": "Kolmogorov-Smirnov",
            "statistic": statistic,
            "p_value": p_value,
            "drift_detected": detected
        })

    for column in CATEGORICAL_COLUMNS:
        statistic, p_value, detected = detect_categorical_drift(
            reference_data, current_data, column
        )
        detailed_results.append({
            "feature": column,
            "test": "Chi-square",
            "statistic": statistic,
            "p_value": p_value,
            "drift_detected": detected
        })

    pd.DataFrame(detailed_results).to_csv(
        PROJECT_ROOT / "monitoring" / "drift_results.csv", index=False
    )

    print("Multiple drift-level experiments completed.")
    print(pd.DataFrame(experiment_results).to_string(index=False))
    print(f"Results saved at: {output_path}")


if __name__ == "__main__":
    main()
