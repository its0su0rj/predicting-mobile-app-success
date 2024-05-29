import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Set up the main title
st.title("Predicting App Success")

# Set up the sidebar with the app features
st.sidebar.title("App Features")
st.sidebar.header("Ratings")
st.sidebar.slider("Rating", 1, 5, 3)

st.sidebar.header("Price")
st.sidebar.selectbox("Price Range", ["$", "$$", "$$$", "$$$$"])

st.sidebar.header("User Feedback")
st.sidebar.text_area("Feedback")

st.sidebar.header("Prime Genre")
st.sidebar.selectbox("Genre", ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Thriller"])

# Adding three more features
st.sidebar.header("Release Year")
st.sidebar.slider("Year", 1980, 2023, 2000)

st.sidebar.header("Duration")
st.sidebar.slider("Duration (minutes)", 60, 180, 120)

st.sidebar.header("Director")
st.sidebar.text_input("Director Name")

# Main area with model selection and evaluation
st.header("Movie Rating Prediction Models")

# Model selection
model_choice = st.selectbox("Choose a prediction model", ["Support Vector Machine", Random Forest", Decision Tree Classifier"])

# Load dataset for demonstration
@st.cache_data
def load_data():
    data = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# Model training and evaluation
if model_choice == "Support Vector Machine":
    model = SVC()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.header(f"{model_choice}")
st.write(f"Model performance metrics:")
st.write(f"Accuracy: {accuracy:.2f}")

# Example data visualization (placeholder for actual model output)
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='o')
ax.set_title('Model Predictions Visualization')

# Display the plot in the main area
st.pyplot(fig)

# Additional styling and content can be added here
st.markdown("""
<style>
    .main {
        background-color: #f0f0f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0eb;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
