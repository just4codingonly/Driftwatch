import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"

# Model output location
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


# 1. Load dataset
print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {data.shape[0]} records")


# 2. Separate input features and target
X = data.drop(
    columns=["customer_id", "churn"]
)

y = data["churn"]


# 3. Display class distribution
print("\nChurn class distribution:")
print(y.value_counts())


# 4. Identify column types
categorical_columns = [
    "contract_type"
]

numeric_columns = [
    "age",
    "monthly_income",
    "monthly_usage_hours",
    "support_calls"
]


# 5. Preprocess categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numeric_columns
        )
    ]
)


# 6. Create machine-learning pipeline
model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)


# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Train the model
print("\nTraining improved model...")

model_pipeline.fit(
    X_train,
    y_train
)


# 9. Evaluate the model
predictions = model_pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 50)
print("IMPROVED MODEL EVALUATION")
print("=" * 50)

print(f"\nModel accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# 10. Save the trained model
joblib.dump(
    model_pipeline,
    MODEL_PATH
)

print(
    f"\nModel saved successfully at: {MODEL_PATH}"
)