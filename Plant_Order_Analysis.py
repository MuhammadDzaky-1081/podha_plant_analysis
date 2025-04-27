"""
Streamlit Dashboard: Podha Plants Analytics
Dependencies: streamlit, pandas, numpy, scikit-learn, seaborn, matplotlib
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ------------------
# 1. Data & Model Loaders
# ------------------
@st.cache_data
def load_data(path: str = 'podha_plants_order.csv') -> pd.DataFrame:
    """
    Load dataset and parse date columns.
    """
    df = pd.read_csv(path)
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

@st.cache_resource
def load_model(path: str = 'model.pkl'):
    """
    Load ML model from pickle file.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# ------------------
# 2. Page Functions
# ------------------

def page_marketing(df: pd.DataFrame):
    st.header("📈 Analisis Kampanye Pemasaran")
    req_cols = {'campaign', 'cost', 'revenue', 'order_id'}
    if not req_cols.issubset(df.columns):
        st.warning(f"Dataset perlu kolom: {', '.join(req_cols)}")
        return

    summary = (
        df.groupby('campaign')
          .agg(
              total_cost=('cost', 'sum'),
              total_revenue=('revenue', 'sum'),
              orders=('order_id', 'nunique')
          )
          .reset_index()
    )
    summary['profit'] = summary['total_revenue'] - summary['total_cost']
    summary['ROI'] = summary['profit'] / summary['total_cost']

    st.subheader("Ringkasan Kampanye")
    st.dataframe(summary)

    # Plot ROI & Orders
    fig, ax = plt.subplots()
    data_melt = summary.melt(id_vars='campaign', value_vars=['ROI', 'orders'], var_name='Metric')
    sns.barplot(x='campaign', y='value', hue='Metric', data=data_melt, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

def page_product_customer(df: pd.DataFrame):
    st.header("🛒 Segmentasi Produk & Pelanggan")

    if 'product' in df.columns:
        top = df['product'].value_counts().nlargest(10)
        st.subheader("Top 10 Produk")
        fig, ax = plt.subplots()
        sns.barplot(x=top.values, y=top.index, ax=ax)
        ax.set_xlabel('Count')
        st.pyplot(fig)
    else:
        st.warning("Kolom 'product' tidak ditemukan.")

    if {'customer_id', 'order_value'}.issubset(df.columns):
        cust = df.groupby('customer_id').agg(
            orders=('order_id', 'nunique'),
            total_value=('order_value', 'sum')
        )
        kmeans = KMeans(n_clusters=3, random_state=42)
        cust['cluster'] = kmeans.fit_predict(cust)

        st.subheader("Cluster Pelanggan (KMeans)")
        fig, ax = plt.subplots()
        sns.scatterplot(x='orders', y='total_value', hue='cluster', data=cust.reset_index(), palette='deep', ax=ax)
        ax.set_xlabel('Jumlah Orders')
        ax.set_ylabel('Total Nilai Order')
        st.pyplot(fig)
    else:
        st.warning("Kolom 'customer_id' atau 'order_value' hilang.")

def page_finance_risk(df: pd.DataFrame):
    st.header("💰 Analisis Keuangan & Risiko")

    if 'order_value' in df.columns:
        total = df['order_value'].sum()
        st.metric("Total Revenue", f"Rp{total:,.0f}")

    if 'payment_status' in df.columns:
        failed = df[df['payment_status'] != 'success']
        rate = len(failed) / len(df) * 100
        st.metric("Gagal Pembayaran (%)", f"{rate:.2f}%")

    if {'service', 'order_value'}.issubset(df.columns):
        svc = df.groupby('service')['order_value'].sum()
        st.subheader("Pendapatan per Layanan")
        fig, ax = plt.subplots()
        svc.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    else:
        st.warning("Kolom 'service' atau 'order_value' tidak tersedia.")


def page_strategy_forecasting(df: pd.DataFrame):
    st.header("🔭 Proyeksi Pesanan (Naive Average)")
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if not date_cols or 'order_value' not in df.columns:
        st.warning("Butuh kolom tanggal dan order_value untuk forecasting.")
        return
    date_col = date_cols[0]

    ts = df[[date_col, 'order_value']].dropna().set_index(date_col).resample('M').sum()
    last_year = ts['order_value'].last('12M')
    mean_val = last_year.mean()
    future_idx = pd.date_range(start=ts.index.max() + pd.offsets.MonthBegin(), periods=12, freq='M')
    forecast = pd.Series([mean_val]*12, index=future_idx)

    combined = pd.concat([ts['order_value'], forecast.rename('Forecast')], axis=1)
    combined.columns = ['Actual', 'Forecast']

    fig, ax = plt.subplots()
    combined.plot(ax=ax)
    ax.set_ylabel('Order Value')
    st.pyplot(fig)

def page_model_insights(model):
    st.header("🔍 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        fig, ax = plt.subplots()
        sns.barplot(x=imp.values, y=imp.index, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Model tidak memiliki attribute feature_importances_.")

def page_about():
    st.header("ℹ️ Tentang")
    st.markdown(
        """
        **Data:** podha_plants_order.csv  
        **Model:** model.pkl (pickle)  
        **Libraries:** streamlit, pandas, numpy, scikit-learn, seaborn, matplotlib  
        """
    )

# ------------------
# 3. Main Application
# ------------------

def main():
    st.set_page_config(page_title='Podha Plants Analytics', layout='wide')
    df = load_data()
    model = load_model()

    pages = {
        'Marketing Campaigns': page_marketing,
        'Product & Customer': page_product_customer,
        'Finance & Risk': page_finance_risk,
        'Strategy Forecasting': page_strategy_forecasting,
        'Model Insights': lambda: page_model_insights(model),
        'About': page_about
    }

    choice = st.sidebar.radio('Pilih Halaman', list(pages.keys()))
    # Call pages; pages requiring df pass df, others are no-arg
    if choice in ['Marketing Campaigns', 'Product & Customer', 'Finance & Risk', 'Strategy Forecasting']:
        pages[choice](df)
    else:
        pages[choice]()

if __name__ == '__main__':
    main()
