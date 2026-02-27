import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Prediction System")

st.header("Enter Applicant Details")

# User Inputs
no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
loan_term = st.number_input("Loan Term (Months)", 0)
cibil_score = st.number_input("CIBIL Score", 300, 900)
residential_assets_value = st.number_input("Residential Assets Value", 0)
commercial_assets_value = st.number_input("Commercial Assets Value", 0)
luxury_assets_value = st.number_input("Luxury Assets Value", 0)
bank_asset_value = st.number_input("Bank Asset Value", 0)

# Convert categorical to numeric
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Prediction Button
if st.button("Predict Loan Status"):

    input_data = np.array([[no_of_dependents,
                            education,
                            self_employed,
                            income_annum,
                            loan_amount,
                            loan_term,
                            cibil_score,
                            residential_assets_value,
                            commercial_assets_value,
                            luxury_assets_value,
                            bank_asset_value]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")