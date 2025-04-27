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

# --- Data Loading and Validation ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("podha_plants_order.csv")
        # Basic data validation: Check for required columns
        required_columns = ['OrderID', 'OrderDate', 'CustID', 'ProductSKU', 'OrderQuantity', 'ProductPrice', 'ProductCost', 'AcquisitionSource', 'PaymentMethod', 'Fraud']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in the data file.")

        # Data type conversions
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df['CustID'] = pd.to_numeric(df['CustID'], errors='coerce').astype('Int64')  # Handle potential non-numeric CustID
        df['OrderQuantity'] = pd.to_numeric(df['OrderQuantity'], errors='coerce').astype('Int64')
        df['ProductPrice'] = pd.to_numeric(df['ProductPrice'], errors='coerce')
        df['ProductCost'] = pd.to_numeric(df['ProductCost'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error loading or validating data: {e}")
        return pd.DataFrame()

df_orders = load_data()

# --- Model Loading (Optional) ---
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
st.markdown("This dashboard provides insights into your order data.")

# --- Visualizations and Insights ---
def create_visualization(title, data, x_col, y_col, plot_type='bar', **kwargs):
    if not data.empty:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == 'bar':
            sns.barplot(x=x_col, y=y_col, data=data, ax=ax, **kwargs)
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'hist':
            sns.histplot(data[x_col], kde=True, ax=ax, **kwargs)
        elif plot_type == 'line':
            ax.plot(data[x_col], data[y_col], **kwargs)
            plt.xticks(rotation=45, ha='right')
        else:
            st.warning(f"Unsupported plot type: {plot_type}")
            return
        
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)
    else:
        st.warning(f"No data available for {title}")

# --- Marketing Channel Performance ---
create_visualization('Orders by Acquisition Channel', filtered_df, 'AcquisitionSource', 'OrderID', 
                    plot_type='bar', estimator=len)  

# --- Fraud Detection ---
create_visualization('Fraudulent Orders by Payment Method', filtered_df, 'PaymentMethod', 'OrderID', 
                    plot_type='bar', estimator=len)

# --- Best-Selling Products ---
create_visualization('Top 10 Best-Selling Products', filtered_df.groupby('ProductSKU')['OrderQuantity'].sum().sort_values(ascending=False).head(10).reset_index(),
                     'ProductSKU', 'OrderQuantity', plot_type='bar')

# --- Order Value and Profit Analysis ---
if not filtered_df.empty and all(col in filtered_df.columns for col in ['ProductPrice', 'ProductCost', 'OrderQuantity']):
    filtered_df['OrderValue'] = filtered_df['ProductPrice'] * filtered_df['OrderQuantity']
    filtered_df['Profit'] = filtered_df['ProductPrice'] - filtered_df['ProductCost']

    avg_order_value = filtered_df['OrderValue'].mean()
    total_profit = filtered_df['Profit'].sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Order Value", f"${avg_order_value:.2f}")
    col2.metric("Total Profit", f"${total_profit:.2f}")

    create_visualization('Distribution of Order Values', filtered_df, 'OrderValue', None, plot_type='hist')

# --- Temporal Trends (Orders Over Time) ---
create_visualization('Orders Over Time', filtered_df.groupby(filtered_df['OrderDate'].dt.date)['OrderID'].count().reset_index(),
                     'OrderDate', 'OrderID', plot_type='line')
