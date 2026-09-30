# TrustLens

TrustLens is a lightweight portfolio project for detecting suspicious marketplace listings using multimodal signals such as fake reviews, title-image mismatch, duplicate images, price anomalies, and seller risk.

This repository is initialized for a step-by-step build following the project roadmap in the master prompt.

## Quick start

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate`
3. Install dependencies: `python -m pip install --upgrade pip && pip install -r requirements.txt`
4. Start with the data and modeling steps in order.

## Folder layout

- `data/raw/` — raw downloaded data
- `data/processed/` — processed listings and reviews
- `src/` — pipeline code
- `notebooks/` — exploratory notebooks
- `models/` — saved model artifacts
- `reports/` — metrics and results
- `airflow/dags/` — orchestration DAGs
