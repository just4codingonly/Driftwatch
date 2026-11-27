# DriftWatch Updated Files

Copy these files into your existing project:

- dashboard/app.py -> dashboard/app.py
- dashboard/templates/index.html -> dashboard/templates/index.html
- monitoring/detect_drift.py -> monitoring/detect_drift.py

Run from the project root:

```bash
python monitoring/detect_drift.py
python dashboard/app.py
```

Open:

http://127.0.0.1:5000

The dashboard now includes:
- Low, Medium and High drift experiments
- Experiment comparison table
- Download DriftWatch Report button

## Full pipeline

To rebuild everything from scratch, run these from the project root in this order:

```bash
python data/generate_data.py
python models/train_model.py
python monitoring/simulate_drift.py
python monitoring/detect_drift.py
python monitoring/evaluate_model.py
python monitoring/prediction_drift.py
python monitoring/drift_score.py
python monitoring/visualize_drift.py
python dashboard/app.py
```

`monitoring/prediction_drift.py` compares the model's churn probabilities on the
reference and drifted data (KS test) and saves `monitoring/prediction_drift_results.csv`.
It needs no true labels.

## Project report

The full project report (Group 44, BPIT) is in [`docs/`](docs/):

- [DriftWatch_Project_Report.pdf](docs/DriftWatch_Project_Report.pdf)
- [DriftWatch_Project_Report.docx](docs/DriftWatch_Project_Report.docx)

The results in Chapter 9 of the report come from running the pipeline above
(fixed seeds 42 and 100), so they can be reproduced exactly.
