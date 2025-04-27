```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

# Main app

def main():
    df = load_data()
    model = load_model()

    st.title('Podha Plant Orders Dashboard')

    # Sidebar filters
    st.sidebar.header('Filters')
    regions = ['All'] + sorted(df['Region'].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect('Region', regions, default=['All'])
    sources = ['All'] + sorted(df['AcquisitionSource'].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect('Acquisition Source', sources, default=['All'])

    # Apply filters
    df_filtered = df.copy()
    if 'All' not in selected_regions:
        df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]
    if 'All' not in selected_sources:
        df_filtered = df_filtered[df_filtered['AcquisitionSource'].isin(selected_sources)]

    # Correlation heatmap (only numeric)
    st.header('Correlation Heatmap')
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Forecasting section
    st.sidebar.header('Forecasting')
    budget = st.sidebar.number_input('Budget Allocation', min_value=0.0, step=100.0, value=1000.0)
    # Ensure correct shape for prediction
    pred_customers = model.predict(np.array([[budget]]))[0]
    st.sidebar.metric('Predicted Customers', int(pred_customers))

    # Tabs for insights
    tabs = st.tabs(['Overview', 'Campaign Analysis', 'Product Trends', 'Customer Segmentation', 'Forecast'])

    with tabs[0]:
        st.subheader('Overview')
        st.metric('Total Orders', int(df_filtered.shape[0]))
        st.metric('Total Revenue', f"${df_filtered['ProductPrice'].sum():,.2f}")
        st.metric('Average Profit', f"${df_filtered['Profit'].mean():,.2f}")
        orders_daily = df_filtered.groupby('OrderDate').size().rename('Orders')
        st.line_chart(orders_daily)

    with tabs[1]:
        st.subheader('Campaign Performance')
        profit_by_source = df_filtered.groupby('AcquisitionSource')['Profit'].mean().sort_values(ascending=False)
        st.bar_chart(profit_by_source)
        count_by_source = df_filtered['AcquisitionSource'].value_counts()
        st.bar_chart(count_by_source)

    with tabs[2]:
        st.subheader('Top Products')
        top_products = df_filtered.groupby('ProductSKU')['OrderQuantity'].sum().nlargest(10)
        st.bar_chart(top_products)
        st.subheader('Sales Over Time')
        sales_time = df_filtered.groupby('OrderDate')['OrderQuantity'].sum().rename('Quantity')
        st.line_chart(sales_time)

    with tabs[3]:
        st.subheader('Customer Clusters')
        cust = df_filtered.groupby('CustID').agg(
            orders=('OrderID','nunique'),
            spent=('ProductPrice','sum')
        ).reset_index()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42).fit(cust[['orders','spent']])
        cust['Cluster'] = kmeans.labels_
        cluster_counts = cust['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        st.dataframe(cust.head(10))

    with tabs[4]:
        st.subheader('Acquisition Forecast')
        st.metric('Predicted Customers', f"{int(pred_customers)} based on ${budget:,.0f} budget")
        st.write('Prediction derived from linear model:')
        st.line_chart(pd.DataFrame({'Budget':[budget], 'PredictedCustomers':[pred_customers]}).set_index('Budget'))

if __name__ == '__main__':
    main()
```
