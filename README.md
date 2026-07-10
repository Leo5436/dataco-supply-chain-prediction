# dataco-supply-chain-prediction:Late Delivery & Shipping Time Prediction

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
  and time-based features (order month, quarter, day-of-week)
- **Handled real-world data issues**: latin-1 encoding, missing values,
  high-cardinality categorical fields, PII removal

## Results
| Model | ROC-AUC | Recall |
|-------|---------|--------|
| Logistic Regression | 0.692 | 70% |
| Random Forest | 0.701 | 75.2% |
| **XGBoost (best)** | **0.700** | **77.8%** |

Shipping-time regression: RMSE = 1.25 days (XGBoost)

## Tech Stack
Python · Pandas · scikit-learn · XGBoost · Matplotlib · Seaborn
