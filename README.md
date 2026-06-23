# Medical Recommendation System — Learning Notebook

A hands-on notebook for learning **Logistic Regression**, **Random Forest**, and **Gradient Boosting** through a symptom→disease prediction + recommendation project.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook medical_recommendation_system.ipynb
```

Requires internet access on first run only (downloads and caches the dataset into a local `data/` folder). After that, it runs fully offline.

## What's inside

1. Data download + EDA
2. A data-leakage discovery (why a naive train/test split gives misleading 100% accuracy) and the fix
3. Logistic Regression, Random Forest, and Gradient Boosting — trained, timed, and compared on a leak-free split
4. Feature importance comparison across models
5. A noise-robustness stress test that reveals real differences between the three algorithms
6. A short hyperparameter tuning example (`RandomizedSearchCV` + `GroupKFold`)
7. The recommendation layer: predict a disease, then pull description / precautions / medications / diet / workout suggestions
8. Suggested next steps (XGBoost/LightGBM, SHAP, calibration, FastAPI wrapper)

**Runtime:** roughly 1-2 minutes end-to-end on a laptop CPU (Gradient Boosting is the slow part — ~20s to train due to the 41-class one-vs-rest structure).

## Disclaimer

Educational project on a public, simplified dataset. Not a diagnostic tool — outputs are not medical advice.
