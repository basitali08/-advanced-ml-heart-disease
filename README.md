# Advanced Machine Learning Pipeline - Heart Disease Prediction

## Overview
State-of-the-art heart disease prediction model achieving **91.8% accuracy** and **0.96 ROC AUC** using the UCI Heart Disease dataset. Features automated hyperparameter tuning with Optuna and model interpretation with SHAP.

## Key Results
| Model | Accuracy | ROC AUC | CV Score |
|-------|----------|---------|----------|
| **Random Forest (Tuned)** | **91.80%** | **0.960** | **81.33%** |
| Logistic Regression | 85.25% | 0.958 | 82.98% |
| Gradient Boosting (Tuned) | 83.61% | 0.934 | 78.43% |
| SVM (RBF) | 80.33% | 0.895 | 79.33% |

## Techniques Used
- **Feature Engineering**: Polynomial features, interaction terms, domain-specific ratios
- **Hyperparameter Tuning**: Optuna with 20 trials per model (Random Forest, Gradient Boosting)
- **Model Interpretation**: SHAP (SHapley Additive exPlanations) for global and local explanations
- **Cross-Validation**: 10-fold stratified CV for robust evaluation
- **Feature Scaling**: StandardScaler for all models

## Project Structure
```
advanced-ml-xgboost/
├── data/                   # Real UCI Heart Disease dataset
├── src/
│   └── advanced_ml_pipeline.py  # Complete pipeline
├── models/                 # Saved model artifacts
├── results/                # Plots, SHAP analysis, metrics
└── README.md
```

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn shap optuna
python src/advanced_ml_pipeline.py
```

## SHAP Feature Importance
Top features identified by SHAP analysis:
1. Chest Pain Type (cp)
2. Max Heart Rate (thalach)
3. ST Depression (oldpeak)
4. Number of Major Vessels (ca)
5. Exercise Induced Angina (exang)

## What Makes This Project Advanced
- Automated hyperparameter optimization (Optuna)
- Model interpretability (SHAP summary plots, dependence plots)
- Feature engineering with domain knowledge
- Comprehensive model comparison with statistical rigor
- Production-ready model serialization
