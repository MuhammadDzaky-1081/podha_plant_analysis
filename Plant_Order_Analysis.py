import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Podha Plants Order Analysis Dashboard",
    page_icon="🌱",
    layout="wide"
)

# --- Load Data ---
@st.cache_data  # Cache the data loading process
def load_data():
    try:
        df = pd.read_csv("podha_plants_order.csv")
        # Convert 'OrderDate' to datetime
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce') 
        return df
    except FileNotFoundError:
        st.error("File 'podha_plants_order.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

df_orders = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
if not df_orders.empty:
    start_date = st.sidebar.date_input("Start Date", value=df_orders['OrderDate'].min().date())
    end_date = st.sidebar.date_input("End Date", value=df_orders['OrderDate'].max().date())

    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        filtered_df = pd.DataFrame() 
    else:
        # Convert start_date and end_date to Pandas Timestamp
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        filtered_df = df_orders[
            (df_orders['OrderDate'] >= start_date) & 
            (df_orders['OrderDate'] <= end_date)
        ]

        # Product Category Filter (if column exists)
        if 'Product_Category' in filtered_df.columns:
            product_categories = filtered_df['Product_Category'].unique()
            selected_categories = st.sidebar.multiselect(
                "Product Categories", product_categories, default=product_categories
            )
            filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_categories)]
else:
    st.warning("No data available to filter.")
    filtered_df = pd.DataFrame() 


# --- Title and Introduction ---
st.title("🌱 Podha Plants Order Analysis Dashboard")
st.markdown("This interactive dashboard provides insights into Podha Plants' order data, "
            "enabling data-driven decisions for optimizing marketing campaigns, "
            "increasing profitability, and mitigating fraud.")

# --- Marketing Channel Performance ---
st.subheader("📈 Marketing Channel Performance")
if not filtered_df.empty and 'AcquisitionSource' in filtered_df.columns:
    acquisition_counts = filtered_df['AcquisitionSource'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=acquisition_counts.index, y=acquisition_counts.values, ax=ax)
    ax.set_title('Orders by Acquisition Channel')
    ax.set_xlabel('Acquisition Source')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45, ha='right') 
    st.pyplot(fig)
else:
    st.warning("No data available for Marketing Channel Performance or 'AcquisitionSource' column missing.")

# --- Fraud Detection ---
st.subheader("🔍 Fraud Detection")
if not filtered_df.empty and 'PaymentMethod' in filtered_df.columns:
    fraud_by_payment_method = filtered_df['PaymentMethod'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=fraud_by_payment_method.index, y=fraud_by_payment_method.values, ax=ax)
    ax.set_title('Fraudulent Orders by Payment Method')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Number of Orders')
    st.pyplot(fig)
else:
    st.warning("No data available for Fraud Detection or 'PaymentMethod' column missing.")

# --- Best-Selling Products ---
st.subheader("💰 Best-Selling Products")
if not filtered_df.empty and 'ProductSKU' in filtered_df.columns:
    best_selling_products = filtered_df.groupby('ProductSKU')['OrderQuantity'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=best_selling_products.index, y=best_selling_products.values, ax=ax)
    ax.set_title('Top 10 Best-Selling Products')
    ax.set_xlabel('Product SKU')
    ax.set_ylabel('Total Order Quantity')
    plt.xticks(rotation=45, ha='right')  
    st.pyplot(fig)
else:
    st.warning("No data available for Best-Selling Products or 'ProductSKU' column missing.")

# --- Order Value and Profit Analysis ---
st.subheader("💸 Order Value and Profit Analysis")
if not filtered_df.empty and all(col in filtered_df.columns for col in ['ProductPrice', 'ProductCost', 'OrderQuantity']):
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
else:
    st.warning("No data available for Order Value and Profit Analysis or required columns are missing.")

# --- Temporal Trends (Orders Over Time) ---
st.subheader("📅 Temporal Trends in Orders")
if not filtered_df.empty and 'OrderDate' in filtered_df.columns:
    orders_over_time = filtered_df.groupby(filtered_df['OrderDate'].dt.date)['OrderID'].count()  
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(orders_over_time.index, orders_over_time.values)
    ax.set_title('Orders Over Time')
    ax.set_xlabel('Order Date')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.warning("No data available for Temporal Trends or 'OrderDate' column missing.")
