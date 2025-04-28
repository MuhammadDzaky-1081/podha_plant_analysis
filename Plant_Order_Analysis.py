import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

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
# Date range
min_date, max_date = df['OrderDate'].min(), df['OrderDate'].max()
start_date, end_date = st.sidebar.date_input(
    'Order Date Range',
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
# Region & Source
regions = ['All'] + sorted(df['Region'].dropna().unique())
sources = ['All'] + sorted(df['AcquisitionSource'].dropna().unique())
selected_region = st.sidebar.selectbox('Region', regions)
selected_source = st.sidebar.selectbox('Acquisition Source', sources)
# Forecast input
budget_input = st.sidebar.number_input('Budget (USD)', min_value=0.0, value=1000.0, step=100.0)

# Apply filters
df_filtered = df[
    (df['OrderDate'] >= pd.to_datetime(start_date)) &
    (df['OrderDate'] <= pd.to_datetime(end_date))
]
if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['Region'] == selected_region]
if selected_source != 'All':
    df_filtered = df_filtered[df_filtered['AcquisitionSource'] == selected_source]

# Title and KPIs
st.title('🌱 Podha Plants Interactive Dashboard')
k1, k2, k3 = st.columns(3)
k1.metric('Total Orders', df_filtered.shape[0])
k2.metric('Total Revenue', f"${df_filtered['ProductPrice'].sum():,.2f}")
k3.metric('Avg Profit/Order', f"${df_filtered['Profit'].mean():,.2f}")

# Time-series revenue chart
st.subheader('Revenue Over Time')
revenue_ts = (
    df_filtered.set_index('OrderDate')
    .resample('W')['ProductPrice']
    .sum()
    .reset_index()
)
line = alt.Chart(revenue_ts).mark_line(point=True).encode(
    x='OrderDate:T',
    y='ProductPrice:Q',
    tooltip=['OrderDate:T', alt.Tooltip('ProductPrice:Q', format='$,.2f')]
).properties(width=800, height=300)
st.altair_chart(line, use_container_width=True)

# Download filtered data
csv = df_filtered.to_csv(index=False)
st.download_button(
    label='Download Filtered Data',
    data=csv,
    file_name='filtered_podha_orders.csv',
    mime='text/csv'
)

# Tabs for deeper insights
tab1, tab2, tab3 = st.tabs(['By Acquisition Source', 'Top Products', 'Forecast'])

with tab1:
    st.subheader('Profit by Acquisition Source')
    prof_src = df_filtered.groupby('AcquisitionSource')['Profit'].mean().sort_values()
    st.bar_chart(prof_src)
    st.subheader('Order Count by Source')
    cnt_src = df_filtered['AcquisitionSource'].value_counts()
    st.bar_chart(cnt_src)

with tab2:
    st.subheader('Top 10 Products by Quantity')
    top_prod = df_filtered.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
    st.bar_chart(top_prod)

with tab3:
    st.subheader('Customer Forecasting')
    # Use loaded models if available
    if isinstance(models, dict) and 'linear_regression' in models:
        lr_pred = models['linear_regression'].predict([[budget_input]])[0]
        st.metric('Linear Regression Estimate', f"{int(lr_pred)} customers")
    if isinstance(models, dict) and 'random_forest' in models:
        rf_pred = models['random_forest'].predict([[budget_input]])[0]
        st.metric('Random Forest Estimate', f"{int(rf_pred)} customers")
    if not models:
        simple_pred = int(budget_input / 50)
        st.metric('Simple Rule Estimate', f"{simple_pred} customers")

# Footer
st.markdown('---')
st.caption('Updated dashboard with interactive filters, time-series analysis, and model-based forecasting.')
