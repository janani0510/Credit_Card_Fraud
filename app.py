import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚨 Credit Card Fraud Detection 💳")
st.caption("Model: Logistic Regression")
st.write("Enter the transaction details")

time = st.number_input("Transaction Time",value=0.0, step=1.0)
amount = st.number_input("Transaction Amount",value=0.0, step=0.01)

st.subheader("V1 to V28 Features")
v_features = []

for i in range(28):
    value = st.number_input(f"V{i+1}",value=0.0,step=0.01,key=f"v{i}")
    v_features.append(value)

if st.button("Predict"):

    columns = ["Time"] + [f"V{i}" for i in range(1,29)]+["Amount"]

    features = pd.DataFrame([[time] + v_features + [amount]],columns = columns)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]
    st.write(f"Fraud Probability: **{probability*100:.2f}%**")
    st.progress(float(probability))

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction🚨")
    else:
        st.success("✅ Normal Transaction✅")