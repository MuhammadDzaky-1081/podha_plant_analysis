import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('podha_plants_order.csv')

# Load the models
@st.cache(allow_output_mutation=True)
def load_models():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Main function for the Streamlit app
def main():
    st.title("Plant Order Analysis Dashboard")
    st.sidebar.title("Options")

    # Load data and models
    data = load_data()
    models = load_models()

    # Sidebar options
    page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "Predictions"])

    if page == "Data Overview":
        st.header("Data Overview")
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        st.write("### Dataset Information")
        st.write(data.describe())

        st.write("### Missing Values")
        st.write(data.isnull().sum())

    elif page == "Visualizations":
        st.header("Visualizations")
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

        st.write("### Distribution of Orders")
        column = st.selectbox("Select a column to visualize", data.select_dtypes(include=np.number).columns)
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=30)
        st.pyplot(plt)

    elif page == "Predictions":
        st.header("Predictions")
        st.write("### Select a Model")
        model_name = st.selectbox("Choose a model", list(models.keys()))
        model = models[model_name]

        st.write("### Input Features for Prediction")

        # Assuming the model expects specific features
        feature_columns = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names
        user_input = {}
        for feature in feature_columns:
            user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess input (if needed)
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        st.write("### Prediction Result")
        st.write(f"The predicted value is: {prediction[0]}")

if __name__ == "__main__":
    main()
