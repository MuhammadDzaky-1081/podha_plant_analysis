import streamlit as st
import pandas as pd
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('podha_plants_order.csv', parse_dates=['OrderDate'], dayfirst=False)
    df['Profit'] = df['ProductPrice'] - df['ProductCost']
    return df

# Load dataset
df = load_data()

# Sidebar - Filters
st.sidebar.title('Business Filters')
regions = ['All'] + sorted(df['Region'].dropna().unique())
sources = ['All'] + sorted(df['AcquisitionSource'].dropna().unique())
selected_region = st.sidebar.selectbox('Select Region', regions)
selected_source = st.sidebar.selectbox('Select Acquisition Source', sources)

# Apply filters
if selected_region != 'All':
    df = df[df['Region'] == selected_region]
if selected_source != 'All':
    df = df[df['AcquisitionSource'] == selected_source]

# Sidebar - Forecasting input
st.sidebar.title('Forecasting')
budget_input = st.sidebar.number_input('Input Budget (USD)', min_value=0.0, value=1000.0, step=100.0)

# Main Title
st.title('Podha Plants Interactive Business Dashboard')

# KPI Section
st.subheader('Key Performance Indicators')
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric('Total Orders', df.shape[0])
kpi2.metric('Total Revenue', f"${df['ProductPrice'].sum():,.2f}")
kpi3.metric('Avg Profit per Order', f"${df['Profit'].mean():,.2f}")

# Acquisition Source Analysis
st.subheader('Acquisition Source Performance')
profit_by_source = df.groupby('AcquisitionSource')['Profit'].mean().sort_values(ascending=False)
orders_by_source = df['AcquisitionSource'].value_counts()
st.bar_chart(profit_by_source)
st.bar_chart(orders_by_source)

# Best-Selling Products
st.subheader('Top 10 Best-Selling Products')
top_products = df.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
st.bar_chart(top_products)

# Forecast Section
st.subheader('Simple Forecast Based on Budget')
predicted_customers = int(budget_input / 50)  # Assume $50 budget per customer
st.metric('Estimated New Customers', predicted_customers)
st.caption('Note: Forecast is a simplified approximation based on historical budget efficiency.')

# Footer
st.write("\n")
st.caption('Dashboard generated automatically based on Podha business data.')
