import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Streamlit page configuration
st.set_page_config(
    page_title="Podha Plants Order Analysis Dashboard",
    page_icon="🌱",
    layout="wide"
)

# --- Data Loading and Validation ---
@st.cache_data  # Cache the data loading for faster performance
def load_data():
    try:
        df = pd.read_csv("podha_plants_order.csv")
        # Basic data validation: Check for essential columns
        essential_columns = [
            'OrderID', 'OrderDate', 'CustID', 'ProductSKU', 'OrderQuantity',
            'ProductPrice', 'ProductCost', 'AcquisitionSource', 'PaymentMethod'
        ]
        if not all(col in df.columns for col in essential_columns):
            raise ValueError("Missing essential columns in the data file.")
        
        # Data type conversions
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df['CustID'] = pd.to_numeric(df['CustID'], errors='coerce').astype('Int64')
        df['OrderQuantity'] = pd.to_numeric(df['OrderQuantity'], errors='coerce').astype('Int64')
        df['ProductPrice'] = pd.to_numeric(df['ProductPrice'], errors='coerce')
        df['ProductCost'] = pd.to_numeric(df['ProductCost'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading or validating data: {e}")
        return pd.DataFrame()

@st.cache_data
def save_models():
    # Example models for demonstration
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor()

    # Create a dictionary to store only the implemented models
    models = {
        'linear_regression': linear_model,
        'random_forest': rf_model
    }

    # Save the dictionary to model.pkl
    with open("model.pkl", "wb") as file:
        pickle.dump(models, file)

    st.success("Models have been saved to model.pkl")

# Load data
df_orders = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
if not df_orders.empty:
    start_date = st.sidebar.date_input("Start Date", df_orders['OrderDate'].min().date())
    end_date = st.sidebar.date_input("End Date", df_orders['OrderDate'].max().date())

    filtered_df = df_orders[
        (df_orders['OrderDate'].dt.date >= start_date) &
        (df_orders['OrderDate'].dt.date <= end_date)
    ]

    # Product Category Filter (if column exists)
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
st.markdown("This dashboard provides data-driven insights to optimize your business.")

# --- Insights and Visualizations ---
if not filtered_df.empty:
    # 1. Top Marketing Channels
    if 'AcquisitionSource' in filtered_df.columns:
        st.subheader("Top Marketing Channels")
        top_channels = filtered_df['AcquisitionSource'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_channels.index, y=top_channels.values, ax=ax)
        ax.set_title('Orders by Acquisition Channel (Top 5)')
        ax.set_xlabel('Acquisition Source')
        ax.set_ylabel('Number of Orders')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("**Actionable Insight:** Focus marketing efforts on the top-performing channels to maximize customer acquisition.")

    # 2. Fraud Detection by Payment Method
    if 'PaymentMethod' in filtered_df.columns and 'Fraud' in filtered_df.columns:
        st.subheader("Fraud Detection by Payment Method")
        fraud_by_payment = filtered_df[filtered_df['Fraud'] == 'Fraud']['PaymentMethod'].value_counts()
        if not fraud_by_payment.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=fraud_by_payment.index, y=fraud_by_payment.values, ax=ax)
            ax.set_title('Fraudulent Orders by Payment Method')
            ax.set_xlabel('Payment Method')
            ax.set_ylabel('Number of Fraudulent Orders')
            st.pyplot(fig)
            st.markdown("**Actionable Insight:** Investigate and implement security measures for payment methods with high fraud rates.")
        else:
            st.info("No fraudulent orders found in the selected data.")

    # 3. Best-Selling Products
    if 'ProductSKU' in filtered_df.columns:
        st.subheader("Best-Selling Products")
        top_products = filtered_df.groupby('ProductSKU')['OrderQuantity'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_products.index, y=top_products.values, ax=ax)
        ax.set_title('Top 10 Best-Selling Products')
        ax.set_xlabel('Product SKU')
        ax.set_ylabel('Total Order Quantity')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        st.markdown("**Actionable Insight:** Ensure sufficient inventory of top-selling products and consider promotional strategies to further boost sales.")

    # 4. Order Value and Profit Analysis
    if all(col in filtered_df.columns for col in ['ProductPrice', 'ProductCost', 'OrderQuantity']):
        st.subheader("Order Value and Profit Analysis")
        filtered_df['OrderValue'] = filtered_df['ProductPrice'] * filtered_df['OrderQuantity']
        filtered_df['Profit'] = (filtered_df['ProductPrice'] - filtered_df['ProductCost']) * filtered_df['OrderQuantity']

        avg_order_value = filtered_df['OrderValue'].mean()
        total_profit = filtered_df['Profit'].sum()

        col1, col2 = st.columns(2)
        col1.metric("Average Order Value", f"${avg_order_value:.2f}")
        col2.metric("Total Profit", f"${total_profit:.2f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_df['OrderValue'], kde=True, ax=ax)
        ax.set_title('Distribution of Order Values')
        ax.set_xlabel('Order Value')
        ax.set_ylabel('Number of Orders')
        st.pyplot(fig)

    # 5. Temporal Trends in Orders
    st.subheader("Temporal Trends in Orders")
    orders_over_time = filtered_df.groupby(filtered_df['OrderDate'].dt.date)['OrderID'].count()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(orders_over_time.index, orders_over_time.values)
    ax.set_title('Orders Over Time')
    ax.set_xlabel('Order Date')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.warning("No data available for the selected filters.")
