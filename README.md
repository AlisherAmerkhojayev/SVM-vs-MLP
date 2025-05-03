# Comparative Analysis of MLP and SVM Models for Diabetes Prediction

---

## Personal Motivation *(customize this section)*

This study was undertaken to critically explore the trade-offs between **Multilayer Perceptrons (MLPs)** and **Support Vector Machines (SVMs)** for predictive modeling in healthcare, with a specific focus on **Type 2 diabetes classification**. In many data-driven industries—particularly healthcare, finance, and risk assessment—understanding the implications of model architecture, hyperparameter sensitivity, and interpretability is just as vital as maximizing accuracy. This project not only evaluates model performance but also demonstrates how technical modeling decisions align with broader goals like ethical deployment, cost-sensitivity, and domain adaptability. Insights gleaned from this analysis are applicable to any scenario involving imbalanced data, high-stakes decision-making, and the need for explainability.

---

## Project Overview

This work presents a systematic comparison of MLPs and SVMs on the Pima Indians Diabetes dataset. Beyond superficial accuracy comparisons, we delve into:

- **Hyperparameter optimization via grid search**
- **Cross-validation strategies for generalizability**
- **Feature distribution handling through scaling and imputation**
- **Performance diagnostics using ROC-AUC, confusion matrices, and precision-recall tradeoffs**

---

## Dataset Summary

- **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data)
- **Rows**: 768
- **Features**: 8 (numeric)
- **Target**: `Outcome` ∈ {0 (non-diabetic), 1 (diabetic)}
- **Imbalance**: 65% non-diabetic, 35% diabetic

### Feature Types:
- Demographics: `Age`, `Pregnancies`
- Clinical metrics: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`

### Data Cleaning:
- Implausible values (e.g. `0` for `BMI`) treated as missing
- **Median imputation** applied to preserve robustness

---

## Exploratory Data Analysis

- **Glucose** and **BMI** distributions approximated normality post-imputation
- **Insulin** and **Pedigree Function** were skewed; retained due to medical relevance
- Weak correlations between features—suitable for nonlinear models like MLP and kernelized SVM
- Class imbalance addressed via:
  - **SMOTE** for MLP
  - **Class-weight adjustment** for SVM

---

## Data Processing & Model Setup

- **Scaler**: `StandardScaler` for both models
- **Split**: 80% training / 20% testing
- **Imbalance strategies**:
  - `SMOTE` used for MLP to synthetically oversample minority class
  - `class_weight='balanced'` used for SVM to penalize misclassification proportionally

---

## Model 1: Multilayer Perceptron (MLP)

- **Library**: scikit-learn `MLPClassifier`
- **GridSearchCV parameters**:
  - `activation`: [`relu`, `tanh`]
  - `hidden_layer_sizes`: [(5,), (10,), (15,), (15,15)]
  - `alpha`: [0.0001, 0.001]
  - `solver`: [`adam`, `sgd`]
  - `learning_rate_init`: [0.001, 0.01]
- **Best configuration**: `tanh`, (15,15), `alpha=0.0001`, `adam`, `lr=0.01`

**Performance**:
- Validation Accuracy: **84.3%**
- Test Accuracy: **69.3%**
- AUC Score: **0.72**

> MLP learned flexible representations but showed signs of **overfitting**. Synthetic data from SMOTE likely introduced variance that didn't generalize well.

---

## Model 2: Support Vector Machine (SVM)

- **Library**: scikit-learn `SVC`
- **GridSearchCV parameters**:
  - `kernel`: [`linear`, `poly`, `rbf`]
  - `C`: [0.1, 1, 10]
  - `gamma`: [`scale`, `auto`]
  - `degree`: [3, 4] (for polynomial kernel)
- **Best configuration**: `rbf`, `C=1`, `gamma=scale`, `class_weight='balanced'`

**Performance**:
- Validation Accuracy: **76.9%**
- Test Accuracy: **72.3%**
- AUC Score: **0.79**

> The **RBF kernel** captured complex patterns and **generalized better** than MLP. It achieved higher recall and F1—valuable for early disease detection.

---

## Evaluation Summary

| Model | Val Acc | Test Acc | AUC  | Precision | Recall | F1   | TP | FN | FP | TN |
|-------|---------|----------|------|-----------|--------|------|----|----|----|----|
| MLP   | 84.3%   | 69.3%    | 0.72 | 0.55      | 0.67   | 0.60 | 51 | 25 | 42 | 111 |
| SVM   | 76.9%   | 72.3%    | 0.79 | 0.57      | 0.78   | 0.66 | 62 | 18 | 46 | 105 |

---

## Visualizations

- **ROC curves** compare classifier discrimination ability
- **Confusion matrices** highlight Type I/II error tradeoffs
- **GridSearch heatmaps** show hyperparameter sensitivity

---

## Insights & Applications

- **MLP** is suited for large-scale, rich-feature domains like marketing or credit scoring.
- **SVM** is preferred for structured, small datasets with interpretability demands, such as healthcare, fraud detection, or regulatory systems.
- **False negatives are costlier** in medical contexts → SVM’s better recall is a critical asset.

---

## Future Work

- Add **regularization (dropout)** and **early stopping** to MLP
- Test ensemble learning: **VotingClassifier** combining MLP and SVM
- Use **SHAP** or **LIME** for post-hoc model interpretability
- Apply to **multimodal health records** or **longitudinal clinical datasets**
- Optimize hyperparameters using **Bayesian optimization**

---

## Conclusion

- **MLP** showed high potential but also instability without sufficient data or regularization.
- **SVM** offered more **predictive robustness** and may be preferred in medical or risk-sensitive applications.
- Choice of model should balance **data size, complexity, interpretability**, and **domain-specific constraints**.
