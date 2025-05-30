import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('artifacts/bangalore_house_price_model.pkl')
area_encoder = joblib.load('artifacts/area_type_encoder.pkl')
location_encoder = joblib.load('artifacts/location_encoder.pkl')

# Streamlit UI
st.title("🏠 House Price Prediction")

# Inputs from user
area_type = st.selectbox("Select Area Type", area_encoder.classes_)
location = st.selectbox("Select Location", location_encoder.classes_)
total_sqft = st.number_input("Total Square Feet", min_value=500.0, max_value=10000.0, step=100.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=3, step=1)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=3, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=4, step=1)

# Prepare input for prediction
if st.button("Predict Price"):
    area_encoded = area_encoder.transform([area_type])[0]
    loc_encoded = location_encoder.transform([location])[0]

    input_features = np.array([[area_encoded, loc_encoded, total_sqft, bath, balcony, bhk]])
    prediction = model.predict(input_features)[0]

    st.success(f"🏡 Estimated Price: ₹ {round(prediction, 2)} Lakhs")
