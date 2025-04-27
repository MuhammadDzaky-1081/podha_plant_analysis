import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Podha Plants Order Analysis Dashboard",
    page_icon="🌱",
    layout="wide"
)

# --- Load Data and Models ---
@st.cache_data
def load_data():
    df = pd.read_csv("podha_plants_order.csv")
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')  # Ensure proper datetime parsing
    df = df.dropna(subset=['OrderDate'])  # Drop rows with invalid or missing dates
    df['OrderDate'] = df['OrderDate'].dt.date  # Convert to date format for compatibility with date_input
    return df

@st.cache_resource
def load_models():
    return pickle.load(open("model.pkl", "rb"))

df_orders = load_data()
models = load_models()
linear_model = models['linear_regression']
rf_model = models['random_forest']

# --- Sidebar Filters ---
start_date = st.sidebar.date_input("Start Date", value=df_orders['OrderDate'].min())
end_date = st.sidebar.date_input("End Date", value=df_orders['OrderDate'].max())
filtered_df = df_orders[(df_orders['OrderDate'] >= start_date) & (df_orders['OrderDate'] <= end_date)]

product_categories = filtered_df['Product_Category'].unique()
selected_categories = st.sidebar.multiselect("Product Categories", product_categories, default=product_categories)
filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_categories)]

# --- Title and Introduction ---
st.title("🌱 Podha Plants Order Analysis Dashboard")
st.markdown("""
This interactive dashboard empowers stakeholders to gain valuable insights into Podha Plants' order data, enabling data-driven decisions for optimizing marketing campaigns, increasing profitability, and mitigating fraud.
""")

# --- Customer Segmentation ---
st.header("📊 Customer Segmentation")
st.markdown("""
Understanding customer behavior is crucial for targeted marketing. This section segments customers using Recency, Frequency, and Monetary Value (RFM) analysis.
""")

rfm_data = filtered_df.groupby('CustID').agg(
    Recency=('OrderDate', lambda x: (datetime.now() - pd.to_datetime(x.max())).days),
    Frequency=('OrderID', 'count'),
    MonetaryValue=('ProductPrice', 'sum')
).reset_index()

# Ensure all columns used in the plot contain valid numeric values
rfm_data['Recency'] = pd.to_numeric(rfm_data['Recency'], errors='coerce')
rfm_data['Frequency'] = pd.to_numeric(rfm_data['Frequency'], errors='coerce')
rfm_data['MonetaryValue'] = pd.to_numeric(rfm_data['MonetaryValue'], errors='coerce')

# Drop rows with invalid or missing values
rfm_data = rfm_data.dropna(subset=['Recency', 'Frequency', 'MonetaryValue'])

# Ensure all values are non-negative
rfm_data = rfm_data[(rfm_data['Recency'] >= 0) & (rfm_data['Frequency'] >= 0) & (rfm_data['MonetaryValue'] >= 0)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Recency', y='MonetaryValue', data=rfm_data, hue='Frequency', size='Frequency', ax=ax, palette="viridis")
ax.set_title('RFM Customer Segmentation')
st.pyplot(fig)

st.markdown("""
**Business Value:** Identify high-value customers (high frequency, high monetary value, low recency) for targeted marketing and retention efforts.
""")

# --- Marketing Channel Performance ---
st.header("📈 Marketing Channel Performance")
st.markdown("""
Evaluate marketing channel effectiveness to optimize budget allocation. This section analyzes acquisition sources and conversion rates.
""")

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
st.header("🔍 Fraud Detection")
st.markdown("""
All orders in the dataset were initially flagged as fraudulent. This requires immediate investigation. This section analyzes patterns in fraudulent orders.
""")

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
st.header("💰 Profit Prediction")
st.markdown("""
Predict profit per order using Linear Regression and Random Forest models. This section allows for interactive prediction based on various factors.
""")

st.sidebar.header("Prediction Input")
product_cost = st.sidebar.number_input("Product Cost", min_value=0.0, value=10.0)
product_price = st.sidebar.number_input("Product Price", min_value=0.0, value=20.0)
order_quantity = st.sidebar.number_input("Order Quantity", min_value=1, value=1)

input_data = pd.DataFrame({
    'ProductCost': [product_cost],
    'ProductPrice': [product_price],
    'OrderQuantity': [order_quantity],
})

linear_prediction = linear_model.predict(input_data)[0]
rf_prediction = rf_model.predict(input_data)[0]

st.write("**Profit Prediction Results:**")
st.metric(label="Linear Regression Prediction", value=f"${linear_prediction:.2f}")
st.metric(label="Random Forest Prediction", value=f"${rf_prediction:.2f}")

st.markdown("""
**Business Value:** Optimize pricing strategies, manage inventory levels, and make informed decisions to maximize profitability.
""")

# --- Additional Insights ---
st.header("📌 Additional Insights")
st.markdown("""
Explore additional business aspects such as seasonal trends, top-performing products, and customer loyalty.
""")

# Seasonal Trends
st.subheader("📅 Seasonal Trends")
filtered_df['OrderMonth'] = pd.to_datetime(filtered_df['OrderDate']).dt.to_period('M')
monthly_sales = filtered_df.groupby('OrderMonth')['OrderID'].count()
fig, ax = plt.subplots(figsize=(10, 6))
monthly_sales.plot(kind='line', ax=ax, marker='o', color='green')
ax.set_title('Monthly Sales Trends')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Orders')
st.pyplot(fig)

# Top-Performing Products
st.subheader("🏆 Top-Performing Products")
top_products = filtered_df.groupby('Product_Name')['OrderID'].count().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
top_products.plot(kind='bar', ax=ax, color='orange')
ax.set_title('Top 10 Products by Sales')
ax.set_xlabel('Product Name')
ax.set_ylabel('Number of Orders')
st.pyplot(fig)

# Customer Loyalty
st.subheader("🤝 Customer Loyalty")
loyalty_data = filtered_df.groupby('CustID')['OrderID'].count().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
loyalty_data.plot(kind='bar', ax=ax, color='purple')
ax.set_title('Top 10 Loyal Customers')
ax.set_xlabel('Customer ID')
ax.set_ylabel('Number of Orders')
st.pyplot(fig)

st.markdown("""
**Business Value:** Leverage insights into seasonal trends, top-performing products, and customer loyalty to enhance business strategies.
""")

# --- Conclusion ---
st.markdown("""
This dashboard provides valuable insights into Podha Plants' order data. By leveraging these insights, stakeholders can make informed decisions to optimize marketing campaigns, increase profitability, and mitigate fraud.
""")
