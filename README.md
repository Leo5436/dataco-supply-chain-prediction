# DataCo Supply Chain: Late Delivery & Shipping Time Prediction
Predicting late-delivery risk and actual shipping duration using machine learning on DataCo Global's real-world supply chain dataset (180K+ orders).

## Overview
- **Classification**: predict whether an order will be delivered late
- **Regression**: forecast actual shipping days

## Dataset
[DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
— real transactional data from DataCo Global: 180K+ orders, 50+ features across
provisioning, production, sales, and distribution.

## Key Engineering Decisions
- **Data leakage prevention**: removed `Delivery Status` and post-hoc fields
  that would not be available at prediction time
- **Feature engineering**: created discount-quantity interaction, unit price,
  and time-based features (order month, quarter, day-of-week) for the classification task
- **Handled real-world data issues**: latin-1 encoding, missing values,
  high-cardinality categorical fields, PII removal

## Results
| Model | ROC-AUC | Recall |
|-------|---------|--------|
| Logistic Regression | 0.70 | 59% |
| Random Forest | 0.752 | 56% |
| **XGBoost (best)** | **0.778** | **59%** |

Recall reported for the positive class (late delivery). XGBoost achieved the
best ROC-AUC and overall discrimination.

### Regression — Shipping Days
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 1.39 | 1.13 | 0.270 |
| **Random Forest (best)** | **1.21** | **0.94** | **0.445** |
| XGBoost | 1.25 | 1.00 | 0.405 |

Best model predicts actual shipping duration within ~0.94 days on average (MAE).

## Tech Stack
Python · Pandas · Numpy · scikit-learn · XGBoost · Matplotlib · Seaborn
