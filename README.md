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
