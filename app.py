import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="🍷 Wine Quality Predictor", page_icon="🍇")

# -------------------------------------------------------------
# CUSTOM STYLE (Pink Background)
# -------------------------------------------------------------
st.markdown("""
    <style>
    body {
        background-color: #ffe6f2;  /* light pink background */
        color: #2e2e2e;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #ffe6f2;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #fff;
        color: #000;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #ff4da6;  /* hot pink button */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff66b3;  /* lighter pink on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# TITLE
# -------------------------------------------------------------
st.title("💖 Wine Quality Prediction (Pink Theme)")
st.write("Enter the wine details below and click **Predict** to see the wine quality score.")

# -------------------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------------------
try:
    with open("finalized_RFmodel.sav", "rb") as file:
        RF_model = pickle.load(file)
    with open("scaler_model.sav", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model or Scaler file not found! Make sure `.sav` files are in the same folder.")
    st.stop()

# -------------------------------------------------------------
# INPUT FIELDS
# -------------------------------------------------------------
features = {
    'fixed acidity': (4.0, 15.0, 7.4),
    'volatile acidity': (0.1, 1.5, 0.70),
    'citric acid': (0.0, 1.0, 0.00),
    'residual sugar': (0.5, 15.0, 1.9),
    'chlorides': (0.01, 0.2, 0.07),
    'free sulfur dioxide': (1.0, 70.0, 15.0),
    'total sulfur dioxide': (6.0, 200.0, 46.0),
    'density': (0.990, 1.005, 0.996),   # ✅ properly closed
    'pH': (2.5, 4.5, 3.3),
    'sulphates': (0.3, 2.0, 0.65),
    'alcohol': (8.0, 15.0, 10.0)
}

st.subheader("🧾 Input Wine Details")
input_data = {}
for feature, (min_val, max_val, default) in features.items():
    input_data[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, value=default, step=0.01)

input_df = pd.DataFrame([input_data])

# -------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------
if st.button("🔮 Predict"):
    scaled_input = scaler.transform(input_df)
    prediction = RF_model.predict(scaled_input)
    predicted_quality = int(np.round(prediction[0]))

    st.subheader("🎯 Prediction Result")
    if predicted_quality <= 4:
        st.error(f"❌ Poor Quality Wine — Score: {predicted_quality}")
    elif predicted_quality <= 6:
        st.warning(f"⚠️ Average Quality Wine — Score: {predicted_quality}")
    else:
        st.success(f"✅ Excellent Wine Quality — Score: {predicted_quality}")
        st.balloons()
