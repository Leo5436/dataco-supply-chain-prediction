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
  high-cardinality categorical fields （>50 unique values), PII removal
- **Scaling**: standardized features for Logistic Regression only (tree-based
  models are scale-invariant)
  
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


## Key Takeaways
- **No single model wins across tasks**: XGBoost led on classification
  (ROC-AUC 0.778) while Random Forest led on regression (R² 0.445), showing
  model choice should follow the task and features.
- **The classifiers lean toward predicting on-time** (recall ~0.59 on the late
  class). Since a missed late delivery is costlier than a false alarm, lowering
  the decision threshold below 0.5 would trade precision for higher recall — a
  deliberate business tradeoff.
- **XGBoost regression used default hyperparameters**; tuning (max_depth,
  learning_rate, n_estimators) would likely close the gap with Random Forest.
  
## Tech Stack
Python · Pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn
