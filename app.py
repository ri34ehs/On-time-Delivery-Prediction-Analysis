import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Shipment Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

# --- Load Model ---
MODEL_PATH = os.getenv("MODEL_PATH", "late_shipment_predictor.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure it exists.")
    st.stop()

# --- App Title and Description ---
st.title("📦 AI-Powered Shipment Delay Predictor")
st.markdown(
    "This app predicts whether an e-commerce shipment will be late based on its characteristics."
    " This is a Proof of Concept based on the model we built."
)

# --- User Input Section ---
st.header("Enter Shipment Details")

col1, col2 = st.columns(2)

with col1:
    cost_of_the_product = st.number_input("Cost of Product ($)", min_value=50, max_value=350, value=150)
    discount_offered = st.slider("Discount Offered (%)", min_value=1, max_value=65, value=10)
    customer_care_calls = st.selectbox("Customer Care Calls", [2, 3, 4, 5, 6, 7])
    customer_rating = st.selectbox("Customer Rating", [1, 2, 3, 4, 5])

with col2:
    weight_in_gms = st.number_input("Weight (grams)", min_value=1000, max_value=7000, value=2500)
    prior_purchases = st.slider("Prior Purchases", min_value=2, max_value=10, value=3)
    product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])
    gender = st.selectbox("Gender", ["M", "F"])

st.subheader("Logistics Information")
col3, col4 = st.columns(2)

with col3:
    warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])

with col4:
    mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Road", "Ship"])

# --- Prediction Logic ---
if st.button("Predict Delay Risk", type="primary"):
    input_data = pd.DataFrame({
        'customer_care_calls': [customer_care_calls],
        'customer_rating': [customer_rating],
        'cost_of_the_product': [cost_of_the_product],
        'prior_purchases': [prior_purchases],
        'discount_offered': [discount_offered],
        'weight_in_gms': [weight_in_gms],
        'warehouse_block_B': [1 if warehouse_block == 'B' else 0],
        'warehouse_block_C': [1 if warehouse_block == 'C' else 0],
        'warehouse_block_D': [1 if warehouse_block == 'D' else 0],
        'warehouse_block_F': [1 if warehouse_block == 'F' else 0],
        'mode_of_shipment_Road': [1 if mode_of_shipment == 'Road' else 0],
        'mode_of_shipment_Ship': [1 if mode_of_shipment == 'Ship' else 0],
        'product_importance_low': [1 if product_importance == 'low' else 0],
        'product_importance_medium': [1 if product_importance == 'medium' else 0],
        'gender_M': [1 if gender == 'M' else 0]
    })

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ The shipment is likely to be ON TIME.")
        st.metric(label="Probability of On-Time Delivery", value=f"{prediction_proba[1]:.2%}")
    else:
        st.error("🚨 The shipment has a HIGH RISK of being LATE.")
        st.metric(label="Probability of Being Late", value=f"{prediction_proba[0]:.2%}")
