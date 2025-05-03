# Comparative Analysis of MLP and SVM Models for Diabetes Prediction

---

## Personal Motivation 

This project was designed to explore the comparative performance of two foundational machine learning algorithms—**Multilayer Perceptron (MLP)** and **Support Vector Machine (SVM)**—in the context of **medical diagnostics**, a field where prediction accuracy directly translates into lives impacted. My objective was to gain practical insights into model behavior on small, imbalanced datasets and to understand how preprocessing, hyperparameter tuning, and evaluation methods affect real-world outcomes. The ability to generalize findings from a clinical dataset to other domains such as finance (e.g., fraud detection) or marketing (e.g., churn prediction) illustrates the versatility and transferable value of these approaches.

---

## 📌 Project Overview

This study compares MLP and SVM for **binary classification of Type 2 diabetes** using a dataset from the National Institute of Diabetes and Digestive and Kidney Diseases. Each model was trained and optimized through GridSearchCV with cross-validation. Special attention was given to managing **imbalanced class distributions**, which frequently occur in clinical datasets.

---

## 📊 Dataset Summary

- **Source**: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data)
- **Rows**: 768 female patients (Pima Indian heritage, age > 21)
- **Target variable**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)
- **Attributes**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

> **Class Distribution**: 268 positive (diabetic), 500 negative

### Handling Missing Values
Some features contained physiologically impossible values (e.g. zero BMI or insulin). These were replaced with **NaN**, and **column medians** were imputed to preserve distribution integrity without introducing external bias.

---

## 📊 Exploratory Data Analysis (EDA)

- **Distribution**: Glucose, BloodPressure, and BMI were roughly normal. Others were right-skewed.
- **Outliers**: Retained due to clinical relevance. Outlier patients may provide critical insights in predictive modeling.
- **Correlation Matrix**:
  - Most features showed weak pairwise correlation.
  - Higher correlation observed between SkinThickness and BMI; Age and Pregnancies.

> This lack of strong linear correlation supports the use of models capable of capturing nonlinear relationships—justifying the use of both MLP and kernel-based SVMs.

---

## ⚙️ Methodology

### Data Preprocessing
- **Normalization**: All features scaled to unit norm using `StandardScaler`.
- **Split**: 80% training / 20% testing
- **Class Imbalance Handling**:
  - MLP: SMOTE (Synthetic Minority Oversampling Technique)
  - SVM: `class_weight='balanced'`

> Class imbalance is often overlooked, yet in high-stakes applications like medical screening, failing to detect minority class instances can have severe consequences. This choice reflects a real-world focus on fairness and sensitivity.

---

## 🧠 Model 1: Multilayer Perceptron (MLP)

- **Framework**: `MLPClassifier` from `scikit-learn`
- **GridSearch Parameters**:
  - `activation`: `relu`, `tanh`
  - `hidden_layer_sizes`: [(5,), (10,), (15,), (15,15)]
  - `alpha`: [0.0001, 0.001]
  - `solver`: `adam`, `sgd`
  - `learning_rate_init`: [0.001, 0.01]
- **Validation CV**: 5-fold

> 🏆 **Best Params**: `tanh`, 2 layers of 15 neurons, `alpha=0.0001`, `adam`, `learning_rate_init=0.01`

- **Validation Accuracy**: 84.3%
- **Test Accuracy**: 69.3%
- **AUC Score**: 0.72

### Observations
- MLP demonstrated strong learning capacity, especially after SMOTE balanced the dataset.
- However, the gap between validation and test accuracy hints at **overfitting**, a known risk in MLPs on small data.
- In practice, neural networks require more data and regularization techniques (dropout, early stopping) to generalize well.

> **Applications Elsewhere**: MLP's flexibility makes it valuable in tasks like fraud detection, recommendation systems, or credit scoring—where nonlinearities and interactions are common.

---

## 🧪 Model 2: Support Vector Machine (SVM)

- **Framework**: `SVC` from `scikit-learn`
- **GridSearch Parameters**:
  - `kernel`: `linear`, `poly`, `rbf`
  - `C`: [0.1, 1, 10]
  - `gamma`: `scale`, `auto`
  - `degree`: [3, 4] (for `poly`)
- **Class Imbalance**: `class_weight='balanced'`
- **Validation CV**: 5-fold

> 🏆 **Best Params**: `rbf`, `C=1`, `gamma=scale`, `degree=3`

- **Validation Accuracy**: 76.9%
- **Test Accuracy**: 72.3%
- **AUC Score**: 0.79

### Observations
- SVM outperformed MLP in **generalization** and **recall**—making it better suited for this health-related task.
- Kernel trick allowed it to model complex boundaries without requiring deep architectures.

> **Applications Elsewhere**: SVMs are excellent in domains with small-to-medium data and high precision needs—like document classification, image recognition, and spam detection.

---

## 🔍 Key Evaluation Metrics

| Model | Validation Acc | Test Acc | AUC  | TP | FN | FP | TN |
|-------|----------------|----------|------|----|----|----|----|
| MLP   | 0.843          | 0.693    | 0.72 | 51 | 25 | 42 | 111 |
| SVM   | 0.769          | 0.723    | 0.79 | 62 | 18 | 46 | 105 |

> **SVM achieved higher AUC and true positives**, meaning it correctly identified more diabetic patients. In sensitive applications, recall often takes priority over precision.

---

## 📈 Visualizations

- **ROC Curves** for both models plotted
- **Heatmaps** of grid search scores
- **Confusion matrices** analyzed for each model

---

## 🏁 Conclusion

- **MLP** showed high potential but also instability without sufficient data or regularization.
- **SVM** offered more **predictive robustness** and may be preferred in medical or risk-sensitive applications.
- Choice of model should balance **data size, complexity, interpretability**, and **domain-specific constraints**.
