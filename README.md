# Automated Credit Card Approval Predictor

A machine learning system that automates credit card approval decisions using logistic regression with hyperparameter optimization, achieving 83.6% accuracy.

## Overview

This project builds an automated credit card approval classifier that mimics real-world banking systems. The model analyzes applicant data to predict approval decisions, reducing manual review time and improving consistency in the approval process.

## Dataset

The dataset is a subset of the [UCI Credit Card Approval Dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval), containing anonymized credit card applications with various applicant attributes. The final column represents the approval decision (target variable).

**Dataset Characteristics:**
- 690 applications
- 15 features (mix of categorical and numerical)
- Binary classification (approved/denied)
- Contains missing values requiring preprocessing

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Replaced "?" placeholders with NaN
- **Imputation Strategy**:
  - Categorical features: Mode imputation
  - Numerical features: Mean imputation
- **Encoding**: One-hot encoding for categorical variables (drop_first=True to avoid multicollinearity)

### 2. Model Development
- **Feature Scaling**: StandardScaler normalization for improved model performance
- **Train-Test Split**: 70-30 split with random_state=77 for reproducibility
- **Base Model**: Logistic Regression classifier

### 3. Hyperparameter Optimization
- **Method**: GridSearchCV with 5-fold cross-validation
- **Parameters Tuned**:
  - `tol`: [0.01, 0.001, 0.0001]
  - `max_iter`: [100, 150, 200]
- **Best Parameters**: max_iter=100, tol=0.01

## Key Results

| Metric | Score |
|--------|-------|
| **Training Accuracy (CV)** | 82.6% |
| **Test Accuracy** | **83.6%** |
| **Confusion Matrix** | [[209, 2], [1, 271]] |

**Performance Highlights:**
- High accuracy on both training and test sets (low overfitting)
- Excellent recall: Only 1 false negative and 2 false positives on training data
- Robust generalization to unseen data

## Technologies Used

- **Python 3.8+**
- **pandas & NumPy**: Data manipulation and numerical operations
- **scikit-learn**:
  - LogisticRegression
  - StandardScaler
  - GridSearchCV
  - train_test_split
  - confusion_matrix

## Installation & Usage

```bash
# Install required packages
pip install pandas numpy scikit-learn

# Run the classifier
python credit_card_approval.py
```

## Project Structure

```
├── cc_approvals.data              # Dataset
├── credit_card_approval.py        # Main classification script
└── README.md                      # Project documentation
```

## Model Pipeline

```
Raw Data
    ↓
Missing Value Imputation
    ↓
One-Hot Encoding
    ↓
Feature Scaling
    ↓
Train-Test Split
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Best Model Selection
    ↓
Final Predictions
```

## Future Improvements

- Add feature importance analysis to identify key approval factors
- Create a confusion matrix visualization
- Develop ROC curve and AUC analysis
- Build a simple web interface for real-time predictions
- Add model interpretability with SHAP values

## Business Impact

This automated system can:
- **Reduce Processing Time**: Instant approval decisions vs. hours of manual review
- **Improve Consistency**: Standardized decision-making criteria
- **Scale Operations**: Handle high application volumes efficiently
- **Minimize Human Error**: Eliminate inconsistencies in manual reviews
- **Cost Savings**: Reduce labor costs associated with manual application processing
