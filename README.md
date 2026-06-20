# ML Feature Engineering - Semiconductor Manufacturing Yield Prediction

This project addresses the challenge of predicting Pass/Fail yield in a semiconductor manufacturing process using feature engineering to cut through 591 raw signals and identify the handful that actually matter - building a leaner, more accurate model while reducing production costs and learning time.

---

### Project Objective

To predict whether a semiconductor unit will pass or fail quality testing, using only the most important signals out of hundreds - making the model practical for real-time use on the manufacturing floor without requiring all 591 sensor readings.

---

### Dataset Summary

- **1,567 production records × 591 features** - each row is a single manufactured unit with sensor readings
- Target variable: **Pass/Fail** (`-1` = Pass, `1` = Fail → recoded to `0` and `1`)
- Key challenges: massive dimensionality (591 features), features with 20%+ missing values, constant-value features, multicollinearity, class imbalance (Pass dominates), and potential outliers

---

### Feature Engineering Pipeline (591 → 21 Features)

| Step | Technique | Features Remaining |
|---|---|---|
| Start | Raw dataset | 591 |
| Drop high-null features | Remove columns with >20% missing values | ~559 |
| Drop zero-variance features | Remove columns where all values are identical | 559 |
| Low Variance Filter | Drop features with variance < 0.01 threshold | ~535 |
| Multicollinearity (VIF) | Remove features with VIF > 5 | Reduced |
| Mutual Information Gain | Drop features with MIG = 0 (no dependency with target) | **~21** |

> PCA (20 components) was additionally applied post-feature selection - capturing **97% of variance** - for model performance enhancement.

---

### Analysis Workflow

- Dropped features with >20% null values; imputed remaining nulls with column mean; removed the `Time` column as non-predictive; recoded target from `-1/1` to `0/1`
- Applied **Low Variance Filter** (threshold=0.01), **VIF** (threshold=5), and **Mutual Information Gain** (drop zero-MIG features) sequentially to reduce 591 features down to ~21
- Visualized reduced features using histograms, box plots, pairplots, and heatmap - found right-skewed distributions, potential outliers, and no strong linear correlations between features
- Applied **SMOTE** after train-test split (70:30, stratified) to fix class imbalance; verified train/test statistical consistency via mean and SD comparison

---

### Model Performance

**Decision Tree Variants (baseline exploration)**

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree (base) | Overfit | - |
| Decision Tree + GridSearchCV | 93% | 71% |
| Decision Tree + PCA (20 components) | 89% | 78% |

**ML Pipeline - All Models with PCA + GridSearchCV (StratifiedKFold, 5 splits)**

| Model | Key Observation |
|---|---|
| Logistic Regression | Balanced precision, recall, F1 |
| KNN | High recall - identifies more true positives |
| SVM | Perfect train accuracy - overfitting |
| Random Forest | Perfect train accuracy - overfitting |
| Gradient Boosting | Perfect train accuracy - overfitting |
| **AdaBoost** ✅ | **Best overall - strong balance across precision, recall, and F1; no overfitting** |

> **AdaBoost** was selected as the final model and saved using **Pickle** for future deployment. The Decision Tree + PCA pipeline was also pickled as a lightweight alternative.

---

### Tools Used

- Python, NumPy, Pandas, Matplotlib, Seaborn
- Feature Selection: `VarianceThreshold`, `variance_inflation_factor` (statsmodels), `mutual_info_classif` (Scikit-learn)
- Dimensionality Reduction: `PCA` (20 components, 97% variance explained)
- Imbalanced Data: `SMOTE` (imbalanced-learn)
- ML Pipeline: `Pipeline` + `GridSearchCV` + `StratifiedKFold`
- Models: Logistic Regression, KNN, SVC, Random Forest, Gradient Boosting, AdaBoost
- Cross Validation: KFold (10 splits), LOOCV, StratifiedKFold (5 splits)
- Model Persistence: `pickle`
- Environment: Google Colab

---

### 🔍 Use Case

Semiconductor manufacturers can deploy this model on the production line to flag units likely to fail quality testing - before they complete the full manufacturing process. This reduces wasted production time, cuts cost per unit, and enables engineers to quickly identify which sensor signals are most predictive of yield excursions, speeding up root cause analysis.
