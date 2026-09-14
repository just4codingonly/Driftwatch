from pathlib import Path

import pandas as pd
from flask import Flask, render_template, send_file
import io


# --------------------------------------------------
# Flask application setup
# --------------------------------------------------

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DRIFT_RESULTS_PATH = PROJECT_ROOT / "monitoring" / "drift_results.csv"
PERFORMANCE_RESULTS_PATH = (
    PROJECT_ROOT / "monitoring" / "performance_results.csv"
)
EXPERIMENT_RESULTS_PATH = PROJECT_ROOT / "monitoring" / "drift_experiments.csv"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_drift_results():
    """
    Load feature drift results from CSV.
    """

    if not DRIFT_RESULTS_PATH.exists():
        print("Drift results file not found.")
        return []

    data = pd.read_csv(DRIFT_RESULTS_PATH)

    results = []

    for _, row in data.iterrows():
        result = {
            "feature": row.get("feature", "Unknown"),
            "test": row.get("test", "Unknown"),
            "statistic": float(row.get("statistic", 0)),
            "p_value": float(row.get("p_value", 1)),
            "drift_detected": bool(row.get("drift_detected", False)),
        }

        results.append(result)

    return results


def load_performance_results():
    """
    Load model performance results from CSV.

    CSV format:

    dataset,accuracy,precision,recall,f1_score
    reference,...
    drift,...
    """

    default_performance = {
        "reference_accuracy": 0.0,
        "drift_accuracy": 0.0,
        "accuracy_change": 0.0,
        "reference_precision": 0.0,
        "drift_precision": 0.0,
        "reference_recall": 0.0,
        "drift_recall": 0.0,
        "reference_f1_score": 0.0,
        "drift_f1_score": 0.0,
    }

    if not PERFORMANCE_RESULTS_PATH.exists():
        print("Performance results file not found.")
        return default_performance

    data = pd.read_csv(PERFORMANCE_RESULTS_PATH)

    if data.empty:
        print("Performance results file is empty.")
        return default_performance

    # Find reference and drift rows
    reference_rows = data[data["dataset"].astype(str).str.lower() == "reference"]
    drift_rows = data[data["dataset"].astype(str).str.lower() == "drift"]

    if reference_rows.empty or drift_rows.empty:
        print("Reference or drift row not found in performance CSV.")
        return default_performance

    reference = reference_rows.iloc[0]
    drift = drift_rows.iloc[0]

    reference_accuracy = float(reference["accuracy"])
    drift_accuracy = float(drift["accuracy"])

    performance = {
        "reference_accuracy": reference_accuracy,
        "drift_accuracy": drift_accuracy,

        # Positive value means accuracy increased.
        # Negative value means accuracy decreased.
        "accuracy_change": drift_accuracy - reference_accuracy,

        "reference_precision": float(reference["precision"]),
        "drift_precision": float(drift["precision"]),

        "reference_recall": float(reference["recall"]),
        "drift_recall": float(drift["recall"]),

        "reference_f1_score": float(reference["f1_score"]),
        "drift_f1_score": float(drift["f1_score"]),
    }

    return performance

    # Convert the first row into a dictionary
    row = data.iloc[0].to_dict()

    performance = {
        "reference_accuracy": float(
            row.get("reference_accuracy", row.get("accuracy_reference", 0))
        ),
        "drift_accuracy": float(
            row.get("drift_accuracy", row.get("accuracy_drift", 0))
        ),
        "accuracy_change": float(
            row.get("accuracy_change", row.get("accuracy_change_value", 0))
        ),
        "reference_precision": float(
            row.get("reference_precision", row.get("precision_reference", 0))
        ),
        "drift_precision": float(
            row.get("drift_precision", row.get("precision_drift", 0))
        ),
        "reference_recall": float(
            row.get("reference_recall", row.get("recall_reference", 0))
        ),
        "drift_recall": float(
            row.get("drift_recall", row.get("recall_drift", 0))
        ),
        "reference_f1_score": float(
            row.get("reference_f1_score", row.get("f1_reference", 0))
        ),
        "drift_f1_score": float(
            row.get("drift_f1_score", row.get("f1_drift", 0))
        ),
    }

    return performance


def calculate_drift_summary(feature_results):
    """
    Calculate total features, drifted features,
    drift score and severity.
    """

    total_features = len(feature_results)

    drifted_features = sum(
        1
        for result in feature_results
        if result["drift_detected"]
    )

    if total_features == 0:
        drift_score = 0.0
    else:
        drift_score = (
            drifted_features / total_features
        ) * 100

    if drift_score >= 70:
        severity = "HIGH DRIFT"
    elif drift_score >= 30:
        severity = "MEDIUM DRIFT"
    else:
        severity = "LOW DRIFT"

    return (
        total_features,
        drifted_features,
        drift_score,
        severity,
    )


def calculate_performance_status(performance):
    """
    Check whether important model performance metrics
    have decreased after drift.
    """

    accuracy_change = (
        performance["drift_accuracy"]
        - performance["reference_accuracy"]
    )

    precision_change = (
        performance["drift_precision"]
        - performance["reference_precision"]
    )

    recall_change = (
        performance["drift_recall"]
        - performance["reference_recall"]
    )

    f1_change = (
        performance["drift_f1_score"]
        - performance["reference_f1_score"]
    )

    changes = [
        accuracy_change,
        precision_change,
        recall_change,
        f1_change,
    ]

    degraded_metrics = sum(
        1 for change in changes if change < -0.05
    )

    if degraded_metrics >= 2:
        return "PERFORMANCE DEGRADATION DETECTED"

    elif degraded_metrics == 1:
        return "PERFORMANCE WARNING"

    return "PERFORMANCE STABLE"



def load_experiment_results():
    """Load low, medium and high drift experiment results."""
    if not EXPERIMENT_RESULTS_PATH.exists():
        return []
    try:
        return pd.read_csv(EXPERIMENT_RESULTS_PATH).fillna("").to_dict("records")
    except Exception as error:
        print(f"Experiment results could not be loaded: {error}")
        return []


@app.route("/download-report")
def download_report():
    """Download a combined CSV report of drift and experiment results."""
    drift = pd.DataFrame(load_drift_results())
    experiments = pd.DataFrame(load_experiment_results())

    output = io.StringIO()
    output.write("DRIFTWATCH REPORT\n\n")
    output.write("FEATURE DRIFT RESULTS\n")
    if drift.empty:
        output.write("No feature drift results available.\n")
    else:
        drift.to_csv(output, index=False)

    output.write("\nMULTIPLE DRIFT-LEVEL EXPERIMENTS\n")
    if experiments.empty:
        output.write("No experiment results available.\n")
    else:
        experiments.to_csv(output, index=False)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="driftwatch_report.csv",
    )


# --------------------------------------------------
# Dashboard route
# --------------------------------------------------

@app.route("/")
def dashboard():
    """
    Display the DriftWatch dashboard.
    """

    print("Loading DriftWatch dashboard...")

    # Load feature drift data
    feature_results = load_drift_results()

    # Calculate drift summary
    (
        total_features,
        drifted_features,
        drift_score,
        severity,
    ) = calculate_drift_summary(feature_results)

    # Load model performance data
    performance = load_performance_results()

    # Calculate performance status
    performance_status = calculate_performance_status(
        performance
    )

    # Data format required by index.html
    reference_performance = {
        "accuracy": performance["reference_accuracy"],
        "precision": performance["reference_precision"],
        "recall": performance["reference_recall"],
        "f1_score": performance["reference_f1_score"],
    }

    drift_performance = {
        "accuracy": performance["drift_accuracy"],
        "precision": performance["drift_precision"],
        "recall": performance["drift_recall"],
        "f1_score": performance["drift_f1_score"],
    }

    accuracy_change = performance["accuracy_change"]

    return render_template(
        "index.html",
        total_features=total_features,
        drifted_features=drifted_features,
        drift_score=drift_score,
        severity=severity,
        feature_results=feature_results,
        reference_performance=reference_performance,
        drift_performance=drift_performance,
        accuracy_change=accuracy_change,
        performance_status=performance_status,
        experiment_results=load_experiment_results(),
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )