# Heart Disease Classification | Binary Classification

Machine learning and deep learning pipeline to predict the presence of heart disease from clinical examination data.

## Problem Statement
Predict whether a patient has **heart disease** (1) or not (0) based on 13 clinical features including age, cholesterol, resting blood pressure, and ECG results.

## Dataset
| Attribute | Detail |
|---|---|
| File | `heart.csv` (Cleveland Heart Disease dataset) |
| Records | 303 patients |
| Features | 13 clinical features |
| Target | `target` (0 = No Disease, 1 = Disease) |
| Class Balance | Roughly balanced |

## Methodology
1. **EDA & Visualization** — Feature distributions, target class balance, correlation heatmap
2. **Preprocessing** — `StandardScaler` + `SimpleImputer` via `ColumnTransformer`
3. **ML Models** — Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, XGBoost
4. **DL Model** — Dense Neural Network with `EarlyStopping`
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
6. **ML vs DL Comparison** — Side-by-side performance table

## Results
| Model | Accuracy |
|---|---|
| Multiple ML models evaluated | Logistic Regression, KNN, SVM, RF, GB, XGBoost |
| **Best ML Model** | Selected automatically by highest accuracy |
| **Deep Learning** | Dense layers with sigmoid output |

## Technologies
`Python` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

## File Structure
```
08_Heart_Disease_Classification/
├── project_notebook.ipynb   # Main notebook
├── heart.csv                # Dataset
└── models/                  # Saved model files
```

## How to Run
```bash
cd 08_Heart_Disease_Classification
jupyter notebook project_notebook.ipynb
```
