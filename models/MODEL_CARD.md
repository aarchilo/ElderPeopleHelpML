# Model Card: Care-Needs Level Prediction

## Overview
Predicts the care-needs level of elderly individuals based on insurance claims data.

## Dataset
- Source: Synthetic / Aggregated health insurance claims
- Features: Age, gender, hospital admissions, cost, medications, etc.

## Model
- Algorithm: Gradient Boosting Classifier
- Accuracy: ~85%
- Fairness Checks: Balanced across gender and region (within ±3%)

## Explainability
- SHAP used to interpret feature impact
- Key features: frailty_score, hospital_adm_6m, age, polypharmacy_count

## Limitations
- Synthetic data, needs validation with real data
- Bias possible if demographics uneven

## Authors
Vidhi, Anisha, Aarchi, Asha (M.Tech AI)
