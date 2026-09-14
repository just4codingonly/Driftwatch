import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "monitoring" / "drift_results.csv"
CHART_PATH = PROJECT_ROOT / "monitoring" / "drift_report.png"


def create_drift_chart():
    results = pd.read_csv(RESULTS_PATH)

    # Convert True/False into numerical status values
    results["drift_status"] = results["drift_detected"].map({
        True: 1,
        False: 0
    })

    # Create readable labels
    results["status_label"] = results["drift_detected"].map({
        True: "Drift Detected",
        False: "No Drift"
    })

    # Create the chart
    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        results["feature"],
        results["drift_status"]
    )

    plt.xlabel("Drift Status")
    plt.ylabel("Features")
    plt.title("DriftWatch - Feature Drift Detection")

    # Make the x-axis readable
    plt.xticks(
        [0, 1],
        ["No Drift", "Drift Detected"]
    )

    plt.xlim(0, 1.2)

    # Add labels to each bar
    for bar, label in zip(bars, results["status_label"]):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center"
        )

    plt.tight_layout()

    # Save chart
    plt.savefig(CHART_PATH, dpi=300)

    print("Improved drift chart created successfully!")
    print(f"Chart saved at: {CHART_PATH}")

    plt.show()


if __name__ == "__main__":
    create_drift_chart()