import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("naive_bayes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("CreditWise Loan Approval Predictor")

# Numerical Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18)
Dependents = st.number_input("Dependents", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
Savings = st.number_input("Savings", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term", min_value=0)
Education_Level = st.selectbox(
    "Education Level",
    [0, 1, 2]
)

# Employment Status
emp = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed", "Other"]
)

Employment_Status_Salaried = 1 if emp == "Salaried" else 0
Employment_Status_Self_employed = 1 if emp == "Self-employed" else 0
Employment_Status_Unemployed = 1 if emp == "Unemployed" else 0

# Marital Status
marital = st.selectbox("Marital Status", ["Married", "Single"])
Marital_Status_Single = 1 if marital == "Single" else 0

# Property Area
property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

Property_Area_Semiurban = 1 if property_area == "Semiurban" else 0
Property_Area_Urban = 1 if property_area == "Urban" else 0

# Gender
gender = st.selectbox("Gender", ["Female", "Male"])
Gender_Male = 1 if gender == "Male" else 0

# Employer Category
employer = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed", "Other"]
)

Employer_Category_Government = 1 if employer == "Government" else 0
Employer_Category_MNC = 1 if employer == "MNC" else 0
Employer_Category_Private = 1 if employer == "Private" else 0
Employer_Category_Unemployed = 1 if employer == "Unemployed" else 0

# Loan Purpose
purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

Loan_Purpose_Car = 1 if purpose == "Car" else 0
Loan_Purpose_Education = 1 if purpose == "Education" else 0
Loan_Purpose_Home = 1 if purpose == "Home" else 0
Loan_Purpose_Personal = 1 if purpose == "Personal" else 0

if st.button("Predict"):

    DTI_Ratio_sq = (Loan_Amount / max(Applicant_Income, 1)) ** 2
    Credit_Score_sq = 700 ** 2
    Applicant_Income_log = np.log1p(Applicant_Income)

    data = pd.DataFrame([[
        Applicant_Income,
        Coapplicant_Income,
        Age,
        Dependents,
        Existing_Loans,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        Education_Level,
        Employment_Status_Salaried,
        Employment_Status_Self_employed,
        Employment_Status_Unemployed,
        Marital_Status_Single,
        Property_Area_Semiurban,
        Property_Area_Urban,
        Gender_Male,
        Employer_Category_Government,
        Employer_Category_MNC,
        Employer_Category_Private,
        Employer_Category_Unemployed,
        Loan_Purpose_Car,
        Loan_Purpose_Education,
        Loan_Purpose_Home,
        Loan_Purpose_Personal,
        DTI_Ratio_sq,
        Credit_Score_sq,
        Applicant_Income_log
    ]], columns=[
        'Applicant_Income',
        'Coapplicant_Income',
        'Age',
        'Dependents',
        'Existing_Loans',
        'Savings',
        'Collateral_Value',
        'Loan_Amount',
        'Loan_Term',
        'Education_Level',
        'Employment_Status_Salaried',
        'Employment_Status_Self-employed',
        'Employment_Status_Unemployed',
        'Marital_Status_Single',
        'Property_Area_Semiurban',
        'Property_Area_Urban',
        'Gender_Male',
        'Employer_Category_Government',
        'Employer_Category_MNC',
        'Employer_Category_Private',
        'Employer_Category_Unemployed',
        'Loan_Purpose_Car',
        'Loan_Purpose_Education',
        'Loan_Purpose_Home',
        'Loan_Purpose_Personal',
        'DTI_Ratio_sq',
        'Credit_Score_sq',
        'Applicant_Income_log'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")