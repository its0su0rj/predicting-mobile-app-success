import streamlit as st

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

# Main area with prediction models
st.title("Movie Rating Prediction Models")

st.header("Support Vector Machine")
st.write("Model performance metrics, visualizations, and other details about SVM.")

st.header("Random Forest")
st.write("Model performance metrics, visualizations, and other details about Random Forest.")

st.header("Decision Tree Classifier")
st.write("Model performance metrics, visualizations, and other details about Decision Tree.")

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

# Example data visualization (placeholder for actual model output)
import matplotlib.pyplot as plt
import numpy as np

# Creating a dummy plot
fig, ax = plt.subplots()
ax.plot(np.random.rand(10))
ax.set_title('Example Model Output')

# Display the plot in the main area
st.pyplot(fig)
