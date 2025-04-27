import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Caching data and model
@st.cache_data
def load_data():
    df = pd.read_csv('podha_plants_order.csv', parse_dates=['OrderDate'], dayfirst=False)
    df['Profit'] = df['ProductPrice'] - df['ProductCost']
    return df

@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Load
df = load_data()
model = load_model()

# Sidebar filters
st.sidebar.header("Filters")
regions = ['All'] + sorted(df['Region'].unique().tolist())
selected_regions = st.sidebar.multiselect('Region', regions, default=['All'])
sources = ['All'] + sorted(df['AcquisitionSource'].unique().tolist())
selected_sources = st.sidebar.multiselect('Acquisition Source', sources, default=['All'])

# Apply filters
df_filtered = df.copy()
if 'All' not in selected_regions:
    df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]
if 'All' not in selected_sources:
    df_filtered = df_filtered[df_filtered['AcquisitionSource'].isin(selected_sources)]

# Forecast input
st.sidebar.header("Forecasting")
budget = st.sidebar.number_input('Budget Allocation', min_value=0.0, step=100.0, value=1000.0)
pred_customers = model.predict(np.array([[budget]]))[0]

# Main title
st.title('Podha Interactive Dashboard')

# Tabs
tabs = st.tabs(["Overview", "Campaign Analysis", "Product Trends", "Customer Segmentation", "Forecast"])

with tabs[0]:  # Overview
    st.header('Key Metrics')
    st.metric('Total Orders', int(df_filtered.shape[0]))
    st.metric('Total Revenue', f"${df_filtered['ProductPrice'].sum():,.2f}")
    st.metric('Average Profit', f"${df_filtered['Profit'].mean():,.2f}")
    orders_daily = df_filtered.groupby('OrderDate').size().rename('Orders')
    st.line_chart(orders_daily)

with tabs[1]:  # Campaign Analysis
    st.header('Campaign Performance')
    profit_by_source = df_filtered.groupby('AcquisitionSource')['Profit'].mean().sort_values(ascending=False)
    st.bar_chart(profit_by_source)
    order_counts = df_filtered['AcquisitionSource'].value_counts()
    st.bar_chart(order_counts)

with tabs[2]:  # Product Trends
    st.header('Top Products')
    top_products = df_filtered.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
    st.bar_chart(top_products)
    st.subheader('Trending Over Time')
    sales_time = df_filtered.groupby('OrderDate')['OrderQuantity'].sum().rename('Quantity')
    st.line_chart(sales_time)

with tabs[3]:  # Customer Segmentation
    st.header('Customer Clusters')
    cust = df_filtered.groupby('CustID').agg(
        orders=('OrderID','nunique'),
        spent=('ProductPrice','sum')
    ).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42).fit(cust[['orders','spent']])
    cust['Cluster'] = kmeans.labels_
    cluster_counts = cust['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    st.dataframe(cust.head(10))

with tabs[4]:  # Forecast
    st.header('Acquisition Forecast')
    st.metric('Predicted Customers', int(pred_customers))
    st.write(f"Based on a budget allocation of ${budget:,.0f}.")

# End of app
