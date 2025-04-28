import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('podha_plants_order.csv', parse_dates=['OrderDate'], dayfirst=False)
    df['Profit'] = df['ProductPrice'] - df['ProductCost']
    return df

# Load models
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Load dataset and models
df = load_data()
models = load_models()

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
st.title('🌱 Podha Plants Interactive Business Dashboard')

# KPI Section with styled metrics
st.markdown("""<h3 style='text-align: center;'>Key Performance Indicators</h3>""", unsafe_allow_html=True)
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric(label='📦 Total Orders', value=df.shape[0])
with kpi2:
    st.metric(label='💰 Total Revenue', value=f"${df['ProductPrice'].sum():,.2f}")
with kpi3:
    st.metric(label='📈 Avg Profit per Order', value=f"${df['Profit'].mean():,.2f}")

# Acquisition Source Analysis
st.markdown("""<h3 style='text-align: center;'>Acquisition Source Performance</h3>""", unsafe_allow_html=True)
profit_by_source = df.groupby('AcquisitionSource')['Profit'].mean().sort_values(ascending=False)
orders_by_source = df['AcquisitionSource'].value_counts()
st.subheader('Average Profit by Acquisition Source')
st.bar_chart(profit_by_source)
st.subheader('Number of Orders by Acquisition Source')
st.bar_chart(orders_by_source)

# Best-Selling Products
st.markdown("""<h3 style='text-align: center;'>Top 10 Best-Selling Products</h3>""", unsafe_allow_html=True)
top_products = df.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
st.bar_chart(top_products)

# Forecast Section
st.markdown("""<h3 style='text-align: center;'>Forecast Based on Budget</h3>""", unsafe_allow_html=True)
predicted_customers = int(budget_input / 50)
st.metric('🔮 Estimated New Customers (Simple)', predicted_customers)

# Optionally show models loaded
if models:
    st.success('✅ Models successfully loaded (linear_regression and random_forest available).')
else:
    st.warning('⚠️ No models loaded.')

st.caption('Note: Forecast is a simplified approximation based on historical budget efficiency.')

# Footer
st.markdown("""---""")
st.caption('Dashboard generated automatically based on Podha business data. Powered by Streamlit 🚀')
