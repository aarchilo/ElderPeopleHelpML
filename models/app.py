# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/gb_baseline.joblib")

st.title("Elderly Care-Needs Level Prediction")

uploaded = st.file_uploader("Upload patient CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    X = df.drop(columns=['patient_id'], errors='ignore')
    preds = model.predict(X)
    df['Predicted_Care_Level'] = preds
    st.write(df)

    # SHAP explainability
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("Feature Importance (SHAP)")
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt)
