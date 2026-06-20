# CreditWise Loan Approval Predictor

## Overview

CreditWise Loan Approval Predictor is an end-to-end Machine Learning application that predicts whether a loan application should be **Approved** or **Rejected** based on an applicant's financial, personal, and credit-related information.

The project automates the traditional loan screening process by leveraging Machine Learning techniques, helping financial institutions make faster, more accurate, and unbiased lending decisions.

---

## Problem Statement

A mid-sized financial company, **SecureTrust Bank**, offers personal and home loans to customers across urban and rural regions of India.

Traditionally, loan officers manually review applications by checking:

* Income proofs
* Employment details
* Credit history
* Savings information
* Existing loans
* Collateral value

This manual process is:

* Time-consuming
* Inconsistent
* Prone to human bias

As a result:

1. Good customers may be rejected, leading to loss of business.
2. High-risk customers may be approved, leading to financial losses.

To solve this challenge, an intelligent Machine Learning system was developed to automatically predict whether a loan should be approved before final verification.

---

## Project Objectives

* Build an automated loan approval prediction system.
* Perform data preprocessing and feature engineering.
* Train and evaluate multiple classification algorithms.
* Compare model performance using classification metrics.
* Deploy the best-performing model using Streamlit.

---

## Dataset Description

Each row represents a loan applicant and contains financial, demographic, and credit-related information.

| Feature            | Description                                  |
| ------------------ | -------------------------------------------- |
| Applicant_Income   | Monthly income of applicant                  |
| Coapplicant_Income | Monthly income of co-applicant               |
| Employment_Status  | Salaried / Self-employed / Unemployed        |
| Age                | Applicant age                                |
| Marital_Status     | Married / Single                             |
| Dependents         | Number of dependents                         |
| Credit_Score       | Credit bureau score                          |
| Existing_Loans     | Number of active loans                       |
| DTI_Ratio          | Debt-to-Income Ratio                         |
| Savings            | Savings balance                              |
| Collateral_Value   | Value of collateral                          |
| Loan_Amount        | Requested loan amount                        |
| Loan_Term          | Loan duration                                |
| Loan_Purpose       | Home / Education / Personal / Business       |
| Property_Area      | Urban / Semiurban / Rural                    |
| Education_Level    | Education qualification                      |
| Gender             | Male / Female                                |
| Employer_Category  | Government / Private / MNC / Self-employed   |
| Loan_Approved      | Target Variable (1 = Approved, 0 = Rejected) |

---

## Data Preprocessing

The following preprocessing techniques were applied:

* Missing value handling
* Label Encoding
* One-Hot Encoding
* Feature Scaling using StandardScaler
* Feature Engineering

### Engineered Features

* DTI_Ratio_sq
* Credit_Score_sq
* Applicant_Income_log

---

## Machine Learning Models Evaluated

The following classification algorithms were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Best Performing Model

**Gaussian Naive Bayes**

The Naive Bayes classifier achieved the best overall performance on the dataset and was selected for deployment.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Encoding
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Selection
10. Streamlit Deployment

---

## Streamlit Application

The application allows users to:

* Enter applicant information
* Provide financial details
* Select employment and loan-related attributes
* Receive instant loan approval predictions

---
🚀 Live Demo
Streamlit Application:

https://creditwise-loan-system-fxl2ld3upu7npfdhzhofza.streamlit.app/

## Repository Structure

```text
Creditwise-Loan-System/
│
├── app.py
├── Creditwise_LoanSystem.ipynb
├── loan_approval_data.csv
├── naive_bayes_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Srihitha234/Creditwise-Loan-System.git
```

Move into the project directory:

```bash
cd Creditwise-Loan-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* XGBoost implementation
* Random Forest optimization
* Hyperparameter tuning
* Loan approval probability score
* Explainable AI (SHAP/LIME)
* Enhanced Streamlit dashboard

---

## Author

**Srihitha Babbili**

Machine Learning & Data Science Enthusiast
