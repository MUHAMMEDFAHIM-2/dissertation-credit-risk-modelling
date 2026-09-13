# Credit Risk Grade Prediction — MSc Dissertation (with Fildata)

Interpretable multiclass classification of LendingClub loan grades (A–G), completed as an MSc Data Science dissertation at the University of Salford, run in industry collaboration with **Fildata**.

## Problem

Predict a loan's credit risk grade (7-class: A–G) from application-level features, in a way that stays interpretable enough for use in a regulated lending context — not just accurate.

## Approach

- Methodology: CRISP-DM
- Three parallel experiments:
  - **NO_RATE** — base features, excluding interest rate
  - **WITH_RATE** — includes interest rate as an upper-bound benchmark (rate is normally set *from* the grade, so this is a ceiling check, not the deployable model)
  - **PCA** — dimensionality-reduced base features
- Four algorithms compared across all three experiments: Logistic Regression, Random Forest, Extra Trees, and Histogram-based Gradient Boosting

## Final model

**Tuned HistGradientBoosting, NO_RATE experiment**

| Metric | Score |
|---|---|
| Macro F1 | 0.640 |
| AUC (One-vs-Rest) | 0.951 |
| Weighted Cohen's Kappa | 0.794 |
| Spearman ρ | 1.00 |

Explainability via SHAP, with additional economic validation of the model's decisions against real-world lending outcomes.

## Why NO_RATE is the deliverable, not WITH_RATE

Interest rate is typically assigned *based on* the credit grade, so including it as a predictor leaks the target. WITH_RATE is kept only as a benchmark ceiling; NO_RATE is the model that's actually usable earlier in a real underwriting pipeline.

## Tech stack

Python, scikit-learn, SHAP, pandas, matplotlib, seaborn

## Data

[LendingClub](https://www.lendingclub.com/) public loan dataset.

## Supervision

Supervised by Taha Mansouri, University of Salford, in collaboration with Fildata.
