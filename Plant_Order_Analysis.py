import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('podha_plants_order.csv', parse_dates=['OrderDate'], dayfirst=False)
    df['Profit'] = df['ProductPrice'] - df['ProductCost']
    return df

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Initialize
df = load_data()
models = load_models()

# Sidebar
st.sidebar.title('Filters')
regions = ['All'] + sorted(df['Region'].dropna().unique())
sources = ['All'] + sorted(df['AcquisitionSource'].dropna().unique())
selected_region = st.sidebar.selectbox('Region', regions)
selected_source = st.sidebar.selectbox('Acquisition Source', sources)

if selected_region != 'All':
    df = df[df['Region'] == selected_region]
if selected_source != 'All':
    df = df[df['AcquisitionSource'] == selected_source]

st.sidebar.title('Forecasting')
budget_input = st.sidebar.number_input('Budget Allocation', min_value=0.0, step=100.0, value=1000.0)

# Main page
st.title('Podha Interactive Dashboard')

# Tabs
overview, campaign, products, customers, forecast = st.tabs(['Overview', 'Campaign Analysis', 'Product Trends', 'Customer Segmentation', 'Forecast'])

with overview:
    st.header('Overview Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Orders', df.shape[0])
    col2.metric('Total Revenue', f"${df['ProductPrice'].sum():,.2f}")
    col3.metric('Avg. Profit', f"${df['Profit'].mean():,.2f}")
    daily_orders = df.groupby('OrderDate').size()
    st.line_chart(daily_orders)

with campaign:
    st.header('Campaign Performance')
    st.bar_chart(df.groupby('AcquisitionSource')['Profit'].mean().sort_values(ascending=False))
    st.bar_chart(df['AcquisitionSource'].value_counts())

with products:
    st.header('Product Trends')
    top_products = df.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
    st.bar_chart(top_products)
    sales_over_time = df.groupby('OrderDate')['OrderQuantity'].sum()
    st.line_chart(sales_over_time)

with customers:
    st.header('Customer Segmentation')
    cust_summary = df.groupby('CustID').agg(orders=('OrderID', 'nunique'), spent=('ProductPrice', 'sum')).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42)
    cust_summary['Cluster'] = kmeans.fit_predict(cust_summary[['orders', 'spent']])
    st.bar_chart(cust_summary['Cluster'].value_counts())
    st.dataframe(cust_summary)

with forecast:
    st.header('Forecast Acquisition')
    if 'linear_regression' in models:
        linear_model = models['linear_regression']
        predicted_customers = linear_model.predict(np.array([[budget_input]]))[0]
        st.metric('Predicted Customers', int(predicted_customers))
    if 'random_forest' in models:
        rf_model = models['random_forest']
        rf_predicted_customers = rf_model.predict(np.array([[budget_input]]))[0]
        st.metric('RF Predicted Customers', int(rf_predicted_customers))

    st.write(f"Forecast based on budget of ${budget_input:,.0f}.")
    st.write("Model source: Saved pickle models.")
