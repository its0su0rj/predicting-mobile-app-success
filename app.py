import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved models
rf_model = joblib.load('saved_models/random_forest_model.pkl')
lgbm_model = joblib.load('saved_models/lgbm_model.pkl')
xgb_model = joblib.load('saved_models/xgb_model.pkl')

# Title of the Streamlit app
st.title("Machine Learning Model Evaluation and Visualization")

# Sidebar for uploading data and selecting options
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.sidebar.header("Choose a model to evaluate")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "LightGBM", "XGBoost"))

# Load the uploaded data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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
    accuracy = accuracy_score(y, y_pred)
    st.write(f"### Accuracy of {model_choice}: {accuracy:.4f}")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Visualize feature importance if applicable
    if hasattr(model, 'feature_importances_'):
        st.write("### Feature Importances")
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
