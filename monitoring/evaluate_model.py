import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DATA_PATH = (
    PROJECT_ROOT / "data" / "customer_data.csv"
)

DRIFT_DATA_PATH = (
    PROJECT_ROOT / "data" / "drift_customer_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "churn_model.joblib"
)

RESULTS_PATH = (
    PROJECT_ROOT / "monitoring" / "performance_results.csv"
)


def calculate_metrics(actual_values, predicted_values):
    """
    Calculate classification performance metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            actual_values,
            predicted_values
        ),

        "precision": precision_score(
            actual_values,
            predicted_values,
            zero_division=0
        ),

        "recall": recall_score(
            actual_values,
            predicted_values,
            zero_division=0
        ),

        "f1_score": f1_score(
            actual_values,
            predicted_values,
            zero_division=0
        )
    }

    return metrics


def main():
    print("Loading model and datasets...")

    model = joblib.load(MODEL_PATH)

    reference_data = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    drift_data = pd.read_csv(
        DRIFT_DATA_PATH
    )

    print("Model and datasets loaded successfully!")

    # Separate original features and target
    X_reference = reference_data.drop(
        columns=["customer_id", "churn"]
    )

    y_reference = reference_data["churn"]

    # Create the same test split used during training
    _, X_reference_test, _, y_reference_test = (
        train_test_split(
            X_reference,
            y_reference,
            test_size=0.2,
            random_state=42,
            stratify=y_reference
        )
    )

    # Prepare drifted dataset
    X_drift = drift_data.drop(
        columns=["customer_id", "churn"]
    )

    y_drift = drift_data["churn"]

    # Generate predictions
    print("\nGenerating predictions...")

    reference_predictions = model.predict(
        X_reference_test
    )

    drift_predictions = model.predict(
        X_drift
    )

    # Calculate metrics
    reference_metrics = calculate_metrics(
        y_reference_test,
        reference_predictions
    )

    drift_metrics = calculate_metrics(
        y_drift,
        drift_predictions
    )

    # Display results
    print("\n" + "=" * 60)
    print("       DRIFTWATCH MODEL PERFORMANCE REPORT")
    print("=" * 60)

    print("\nREFERENCE DATA PERFORMANCE")
    print("-" * 60)

    for metric, value in reference_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nDRIFT DATA PERFORMANCE")
    print("-" * 60)

    for metric, value in drift_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Compare performance
    print("\nPERFORMANCE COMPARISON")
    print("-" * 60)

    for metric in reference_metrics:
        difference = (
            drift_metrics[metric]
            - reference_metrics[metric]
        )

        print(
            f"{metric} change: {difference:.4f}"
        )

    # Save results
    results = pd.DataFrame([
        {
            "dataset": "reference",
            **reference_metrics
        },
        {
            "dataset": "drift",
            **drift_metrics
        }
    ])

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nPerformance results saved at: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()