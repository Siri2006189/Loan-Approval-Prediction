import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Loan Approval Prediction System")

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

df.drop("loan_id", axis=1, inplace=True)

le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])
df["self_employed"] = le.fit_transform(df["self_employed"])
df["loan_status"] = le.fit_transform(df["loan_status"])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.header("Enter Applicant Details")

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

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

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