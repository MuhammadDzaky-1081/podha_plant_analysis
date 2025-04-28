import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from datetime import date

# Page configuration
st.set_page_config(
    page_title='Podha Plants Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('podha_plants_order.csv', parse_dates=['OrderDate'])
    df['ProductPrice'] = pd.to_numeric(df['ProductPrice'], errors='coerce')
    df['ProductCost'] = pd.to_numeric(df['ProductCost'], errors='coerce')
    df['Profit'] = df['ProductPrice'] - df['ProductCost']
    return df

# Load models
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Get data and models
df = load_data()
models = load_models()

# Sidebar filters
st.sidebar.title('Filters & Forecast')

# Compute min/max dates and convert to Python date
ts_min = df['OrderDate'].min()
ts_max = df['OrderDate'].max()
min_date = pd.to_datetime(ts_min).date()
max_date = pd.to_datetime(ts_max).date()

# Date range picker
start_date, end_date = st.sidebar.date_input(
    'Order Date Range',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region & Source selectors
regions = ['All'] + sorted(df['Region'].dropna().unique())
sources = ['All'] + sorted(df['AcquisitionSource'].dropna().unique())
selected_region = st.sidebar.selectbox('Region', regions)
selected_source = st.sidebar.selectbox('Acquisition Source', sources)

# Forecast input
budget_input = st.sidebar.number_input('Budget (USD)', min_value=0.0, value=1000.0, step=100.0)

# Apply filters to DataFrame
df_filtered = df[
    (df['OrderDate'].dt.date >= start_date) &
    (df['OrderDate'].dt.date <= end_date)
]
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['Region'] == selected_region]
if selected_source != 'All':
    df_filtered = df_filtered[df_filtered['AcquisitionSource'] == selected_source]

# Main dashboard layout
st.title('🌱 Podha Plants Interactive Dashboard')
col1, col2, col3 = st.columns(3)
col1.metric('Total Orders', df_filtered.shape[0])
col2.metric('Total Revenue', f"${df_filtered['ProductPrice'].sum():,.2f}")
col3.metric('Avg Profit/Order', f"${df_filtered['Profit'].mean():,.2f}")

# Revenue over time chart
st.subheader('Revenue Over Time')
revenue_ts = (
    df_filtered.set_index('OrderDate')
    .resample('W')['ProductPrice']
    .sum()
    .reset_index()
)
chart = alt.Chart(revenue_ts).mark_line(point=True).encode(
    x=alt.X('OrderDate:T', title='Week'),
    y=alt.Y('ProductPrice:Q', title='Revenue'),
    tooltip=[alt.Tooltip('OrderDate:T', title='Date'), alt.Tooltip('ProductPrice:Q', format='$,.2f', title='Revenue')]
).properties(width=800, height=300)
st.altair_chart(chart, use_container_width=True)

# Download filtered data
csv_data = df_filtered.to_csv(index=False)
st.download_button(
    label='Download Filtered Data',
    data=csv_data,
    file_name='filtered_podha_orders.csv',
    mime='text/csv'
)

# Tabs for detailed analysis
tab1, tab2, tab3 = st.tabs(['Acquisition Source', 'Top Products', 'Forecast'])

with tab1:
    st.subheader('Profit by Acquisition Source')
    profit_by_src = df_filtered.groupby('AcquisitionSource')['Profit'].mean().sort_values()
    st.bar_chart(profit_by_src)
    st.subheader('Orders by Acquisition Source')
    orders_by_src = df_filtered['AcquisitionSource'].value_counts()
    st.bar_chart(orders_by_src)

with tab2:
    st.subheader('Top 10 Products by Quantity')
    top_products = df_filtered.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
    st.bar_chart(top_products)

with tab3:
    st.subheader('Customer Forecasting')
    if isinstance(models, dict) and 'linear_regression' in models:
        lr_pred = models['linear_regression'].predict([[budget_input]])[0]
        st.metric('Linear Regression Estimate', f"{int(lr_pred)} customers")
    if isinstance(models, dict) and 'random_forest' in models:
        rf_pred = models['random_forest'].predict([[budget_input]])[0]
        st.metric('Random Forest Estimate', f"{int(rf_pred)} customers")
    if not isinstance(models, dict):
        simple_est = int(budget_input / 50)
        st.metric('Simple Rule Estimate', f"{simple_est} customers")

# Footer
st.markdown('---')
st.caption('Interactive dashboard with date filters, Altair charts, and model-based forecasting.')
