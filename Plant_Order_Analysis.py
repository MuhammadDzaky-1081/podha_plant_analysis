import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Podha Plants Order Analysis Dashboard",
    page_icon="🌱",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("podha_plants_order.csv")
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df_orders = load_data()

# --- Load Models (if available) ---
try:
    with open("model.pkl", "rb") as file:
        models = pickle.load(file)
except Exception as e:
    st.warning(f"Could not load models from model.pkl: {e}")
    models = None

# --- Sidebar Filters ---
st.sidebar.header("Filters")
if not df_orders.empty:
    start_date = st.sidebar.date_input("Start Date", df_orders['OrderDate'].min().date())
    end_date = st.sidebar.date_input("End Date", df_orders['OrderDate'].max().date())

    filtered_df = df_orders[
        (df_orders['OrderDate'].dt.date >= start_date) &
        (df_orders['OrderDate'].dt.date <= end_date)
    ]

    if 'Product_Category' in filtered_df.columns:
        product_categories = filtered_df['Product_Category'].unique()
        selected_categories = st.sidebar.multiselect(
            "Product Categories", product_categories, default=product_categories
        )
        filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_categories)]
else:
    filtered_df = pd.DataFrame()

# --- Title and Introduction ---
st.title("🌱 Podha Plants Order Analysis Dashboard")
st.markdown("This dashboard provides insights into your order data.")

# --- Marketing Channel Performance ---
if not filtered_df.empty and 'AcquisitionSource' in filtered_df.columns:
    st.subheader("📈 Marketing Channel Performance")
    acquisition_counts = filtered_df['AcquisitionSource'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=acquisition_counts.index, y=acquisition_counts.values, ax=ax)
    ax.set_title('Orders by Acquisition Channel')
    ax.set_xlabel('Acquisition Source')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# --- Fraud Detection ---
if not filtered_df.empty and 'PaymentMethod' in filtered_df.columns:
    st.subheader("🔍 Fraud Detection")
    fraud_by_payment_method = filtered_df['PaymentMethod'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=fraud_by_payment_method.index, y=fraud_by_payment_method.values, ax=ax)
    ax.set_title('Fraudulent Orders by Payment Method')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Number of Orders')
    st.pyplot(fig)

# --- Best-Selling Products ---
if not filtered_df.empty and 'ProductSKU' in filtered_df.columns:
    st.subheader("💰 Best-Selling Products")
    best_selling_products = filtered_df.groupby('ProductSKU')['OrderQuantity'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=best_selling_products.index, y=best_selling_products.values, ax=ax)
    ax.set_title('Top 10 Best-Selling Products')
    ax.set_xlabel('Product SKU')
    ax.set_ylabel('Total Order Quantity')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# --- Order Value and Profit Analysis ---
if not filtered_df.empty and all(col in filtered_df.columns for col in ['ProductPrice', 'ProductCost', 'OrderQuantity']):
    st.subheader("💸 Order Value and Profit Analysis")
    filtered_df['OrderValue'] = filtered_df['ProductPrice'] * filtered_df['OrderQuantity']
    filtered_df['Profit'] = filtered_df['ProductPrice'] - filtered_df['ProductCost']

    avg_order_value = filtered_df['OrderValue'].mean()
    total_profit = filtered_df['Profit'].sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Order Value", f"${avg_order_value:.2f}")
    col2.metric("Total Profit", f"${total_profit:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(filtered_df['OrderValue'], kde=True, ax=ax)
    ax.set_title('Distribution of Order Values')
    ax.set_xlabel('Order Value')
    st.pyplot(fig)

# --- Temporal Trends (Orders Over Time) ---
if not filtered_df.empty and 'OrderDate' in filtered_df.columns:
    st.subheader("📅 Temporal Trends in Orders")
    orders_over_time = filtered_df.groupby(filtered_df['OrderDate'].dt.date)['OrderID'].count()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(orders_over_time.index, orders_over_time.values)
    ax.set_title('Orders Over Time')
    ax.set_xlabel('Order Date')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
