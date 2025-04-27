import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the model dictionary
models = pickle.load(open("model.pkl", "rb"))
linear_model = models['linear_regression']
rf_model = models['random_forest']

# Load the dataset
df_orders = pd.read_csv("podha_plants_order.csv")

# --- Page Configuration ---
st.set_page_config(
    page_title="Podha Plants Order Analysis Dashboard",
    page_icon="🌱",  # You can replace this with a suitable emoji or image URL
    layout="wide"  # Use wide layout for better visualization arrangement
)

# --- Title and Introduction ---
st.title("Podha Plants Order Analysis Dashboard")
st.markdown("""
This interactive dashboard empowers stakeholders to gain valuable insights into Podha Plants' order data, enabling data-driven decisions for optimizing marketing campaigns, increasing profitability, and mitigating fraud.
""")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
# Date Filter
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(df_orders['OrderDate'].min()))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(df_orders['OrderDate'].max()))
filtered_df = df_orders[(pd.to_datetime(df_orders['OrderDate']) >= start_date) & (pd.to_datetime(df_orders['OrderDate']) <= end_date)]

# Product Category Filter
product_categories = filtered_df['Product_Category'].unique()
selected_categories = st.sidebar.multiselect("Product Categories", product_categories, default=product_categories)
filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_categories)]

# --- Customer Segmentation ---
st.header("Customer Segmentation")
st.markdown("""
Understanding customer behavior is crucial for targeted marketing. This section segments customers using Recency, Frequency, and Monetary Value (RFM) analysis.
""")

# RFM Calculation and Visualization
rfm_data = filtered_df.groupby('CustID').agg(
    Recency=('OrderDate', lambda x: (datetime.now() - pd.to_datetime(x.max())).days),
    Frequency=('OrderID', 'count'),
    MonetaryValue=('ProductPrice', 'sum')
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Recency', y='MonetaryValue', data=rfm_data, hue='Frequency', size='Frequency', ax=ax)
ax.set_title('RFM Customer Segmentation')
st.pyplot(fig)

st.markdown("""
**Business Value:** Identify high-value customers (high frequency, high monetary value, low recency) for targeted marketing and retention efforts.
""")

# --- Marketing Channel Performance ---
st.header("Marketing Channel Performance")
st.markdown("""
Evaluate marketing channel effectiveness to optimize budget allocation. This section analyzes acquisition sources and conversion rates.
""")

# Marketing Channel Performance Visualization
acquisition_conversion = filtered_df.groupby('AcquisitionSource')['OrderID'].count() / len(filtered_df)
fig, ax = plt.subplots(figsize=(10, 6))
acquisition_conversion.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Conversion Rate per Acquisition Channel')
ax.set_xlabel('Acquisition Source')
ax.set_ylabel('Conversion Rate')
st.pyplot(fig)

st.markdown("""
**Business Value:** Optimize marketing budget allocation by understanding channel performance and conversion rates.
""")

# --- Fraud Detection ---
st.header("Fraud Detection")
st.markdown("""
All orders in the dataset were initially flagged as fraudulent. This requires immediate investigation. This section analyzes patterns in fraudulent orders.
""")

# Fraud Analysis Visualizations
st.write("**Fraudulent Order Distribution by Payment Method:**")
fig, ax = plt.subplots(figsize=(8, 6))
filtered_df['PaymentMethod'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Fraudulent Orders by Payment Method')
ax.set_xlabel('Payment Method')
ax.set_ylabel('Number of Orders')
st.pyplot(fig)

st.markdown("""
**Business Value:** Identify fraud patterns to implement preventive measures and minimize financial losses.
""")

# --- Profit Prediction ---
st.header("Profit Prediction")
st.markdown("""
Predict profit per order using Linear Regression and Random Forest models. This section allows for interactive prediction based on various factors.
""")

# --- Sidebar for Prediction Input ---
st.sidebar.header("Prediction Input")

# Example Input Features (Customize based on your model)
product_cost = st.sidebar.number_input("Product Cost", min_value=0.0, value=10.0)
product_price = st.sidebar.number_input("Product Price", min_value=0.0, value=20.0)
order_quantity = st.sidebar.number_input("Order Quantity", min_value=1, value=1)
# ... Add other input features ...

# --- Prediction Logic ---
# Create a DataFrame with input features
input_data = pd.DataFrame({
    'ProductCost': [product_cost],
    'ProductPrice': [product_price],
    'OrderQuantity': [order_quantity],
    # ... Add other input features ...
})

# Make predictions
linear_prediction = linear_model.predict(input_data)[0]  # Assuming single prediction
rf_prediction = rf_model.predict(input_data)[0]  # Assuming single prediction

# --- Display Predictions ---
st.write("**Profit Prediction Results:**")
st.write(f"Linear Regression Prediction: {linear_prediction:.2f}")
st.write(f"Random Forest Prediction: {rf_prediction:.2f}")

st.markdown("""
**Business Value:** Optimize pricing strategies, manage inventory levels, and make informed decisions to maximize profitability.
""")

# --- Conclusion ---
st.markdown("""
This dashboard provides valuable insights into Podha Plants' order data. By leveraging these insights, stakeholders can make informed decisions to optimize marketing campaigns, increase profitability, and mitigate fraud.
""")
