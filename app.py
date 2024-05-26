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

# Title of the Streamlit app
st.title("Mobile App Success Prediction")

# Sidebar for uploading data and selecting options
st.sidebar.header("Upload your data")
uploaded_file_description = st.sidebar.file_uploader("Upload your applestore_description.csv file", type=["csv"])
uploaded_file_applestore = st.sidebar.file_uploader("Upload your applestore.csv file", type=["csv"])

st.sidebar.header("Choose a model to evaluate")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "LightGBM", "XGBoost"))

# Load and merge the uploaded data
if uploaded_file_description is not None and uploaded_file_applestore is not None:
    # Read the CSV files
    data_description = pd.read_csv(uploaded_file_description)
    data_applestore = pd.read_csv(uploaded_file_applestore)

    # Merge the data on 'id'
    data_merged = pd.merge(data_description, data_applestore, on='id')

    st.write("### Merged Data Preview")
    st.write(data_merged.head())

    # Extract the features (excluding 'id', 'track_name', and any other non-numeric columns)
    # Assuming 'app_desc' needs to be handled or dropped if it's a text column.
    X = data_merged.drop(columns=['id', 'track_name', 'app_desc', 'currency', 'ver', 'cont_rating', 'prime_genre'])
    y = data_merged['user_rating']  # Assuming 'user_rating' is the target variable

    # Predict using the selected model
    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "LightGBM":
        model = lgbm_model
    else:
        model = xgb_model

    # Make predictions
    y_pred = model.predict(X)

    # Display accuracy
    st.write(f"### Predictions for {model_choice}:")
    data_merged['Predicted Success'] = y_pred
    st.write(data_merged[['track_name', 'Predicted Success']])

    # Plot heatmap of correlations
    st.write("### Heatmap of feature correlations:")
    corr = data_merged.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Plot a count plot of the predictions
    st.write("### Count plot of predictions:")
    fig, ax = plt.subplots()
    sns.countplot(x='Predicted Success', data=data_merged, ax=ax)
    st.pyplot(fig)

else:
    st.write("Please upload both CSV files to proceed.")
