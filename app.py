import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lgbm_model = joblib.load('lgbm_model.pkl')

# Define the prediction function
def predict_app_success(model, features):
    prediction = model.predict([features])
    return "Success" if prediction[0] == 1 else "Not Success"

# Streamlit app
st.title("App Success Predictor")

st.write("""
This app predicts whether a new app will be successful based on its features.
""")

# Input fields
size_bytes_in_MB = st.number_input('Size (in MB)', min_value=0.0, value=100.0)
isNotFree = st.selectbox('Is Not Free?', [0, 1])
price = st.number_input('Price (if not free)', min_value=0.0, value=0.0)
rating_count_before = st.number_input('Rating Count Before', min_value=0, value=0)
sup_devices_num = st.number_input('Supported Devices Number', min_value=0, value=1)
ipadSc_urls_num = st.number_input('iPad Screenshots Number', min_value=0, value=1)
lang_num = st.number_input('Languages Number', min_value=0, value=1)
vpp_lic = st.selectbox('VPP License?', [0, 1])
prime_genre = st.selectbox('Primary Genre', df_app['prime_genre'].unique())

# Preprocess the input
input_features = {
    'size_bytes_in_MB': size_bytes_in_MB,
    'isNotFree': isNotFree,
    'price': price,
    'rating_count_before': rating_count_before,
    'sup_devices.num': sup_devices_num,
    'ipadSc_urls.num': ipadSc_urls_num,
    'lang.num': lang_num,
    'vpp_lic': vpp_lic
}

for genre in df_app['prime_genre'].unique():
    input_features[f'prime_genre_{genre}'] = 1 if genre == prime_genre else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_features])

# Model selection
model_choice = st.selectbox('Choose a Model', ['RandomForest', 'LightGBM', 'XGBoost'])
if model_choice == 'RandomForest':
    model = rf_model
elif model_choice == 'LightGBM':
    model = lgbm_model
else:
    model = xgb_model

# Predict and display
if st.button('Predict'):
    prediction = predict_app_success(model, input_df.iloc[0].values)
    st.write(f"The app is predicted to be: {prediction}")
