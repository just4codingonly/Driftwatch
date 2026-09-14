import pandas as pd
import numpy as np
from pathlib import Path


def generate_customer_data(number_of_records=1000, random_seed=42):
    np.random.seed(random_seed)

    data = {
        "customer_id": range(1, number_of_records + 1),
        "age": np.random.randint(18, 70, number_of_records),
        "monthly_income": np.random.randint(15000, 100000, number_of_records),
        "monthly_usage_hours": np.random.randint(1, 100, number_of_records),
        "support_calls": np.random.randint(0, 10, number_of_records),
        "contract_type": np.random.choice(
            ["Monthly", "Yearly", "Two-Year"],
            number_of_records
        ),
        "churn": np.random.choice(
            [0, 1],
            number_of_records,
            p=[0.75, 0.25]
        )
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_folder = project_root / "data"
    output_folder.mkdir(exist_ok=True)

    dataset = generate_customer_data()

    output_path = output_folder / "customer_data.csv"
    dataset.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print(f"Saved at: {output_path}")
    print(f"Number of records: {len(dataset)}")
    print("\nFirst five rows:")
    print(dataset.head())