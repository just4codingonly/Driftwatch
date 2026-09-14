import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
DRIFT_DATA_PATH = PROJECT_ROOT / "data" / "drift_customer_data.csv"


def create_drift_dataset():
    np.random.seed(100)

    # Load the original dataset
    original_data = pd.read_csv(ORIGINAL_DATA_PATH)

    # Create a copy so the original data remains unchanged
    drift_data = original_data.copy()

    # Simulate changes in customer behavior
    drift_data["age"] = np.random.randint(
        25,
        80,
        len(drift_data)
    )

    drift_data["monthly_income"] = np.random.randint(
        30000,
        150000,
        len(drift_data)
    )

    drift_data["monthly_usage_hours"] = np.random.randint(
        30,
        150,
        len(drift_data)
    )

    drift_data["support_calls"] = np.random.randint(
        3,
        15,
        len(drift_data)
    )

    drift_data["contract_type"] = np.random.choice(
        ["Monthly", "Yearly", "Two-Year"],
        len(drift_data),
        p=[0.65, 0.25, 0.10]
    )

    # Save the drifted dataset
    drift_data.to_csv(DRIFT_DATA_PATH, index=False)

    print("Drift dataset created successfully!")
    print(f"Saved at: {DRIFT_DATA_PATH}")
    print(f"Number of records: {len(drift_data)}")

    print("\nOriginal data summary:")
    print(original_data.describe())

    print("\nDrift data summary:")
    print(drift_data.describe())


if __name__ == "__main__":
    create_drift_dataset()