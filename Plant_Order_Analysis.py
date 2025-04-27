import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Podha Plants Analytics Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2e7d32;
    text-align: center;
}
.sub-header {
    font-size: 1.8rem;
    color: #33691e;
}
.section-header {
    font-size: 1.5rem;
    color: #1b5e20;
    border-bottom: 2px solid #81c784;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.metric-card {
    background-color: #f1f8e9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the orders data
        df_orders = pd.read_csv('podha_plants_order.csv')
        
        # Create placeholder data for other datasets if they don't exist
        # This is just for demonstration - in a real scenario, you'd load actual files
        
        # Marketing campaign data
        campaign_data = {
            'campaign_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'campaign_name': ['Summer Sale', 'New Customer', 'Loyalty Program', 'Holiday Special', 'Spring Collection'],
            'cost': [5000, 7500, 3000, 10000, 4500],
            'users_reached': [50000, 100000, 25000, 120000, 40000],
            'clicks': [7500, 12000, 5000, 18000, 6000],
            'conversions': [375, 480, 500, 540, 360],
            'revenue': [18750, 24000, 30000, 32400, 18000],
            'acquisition_source': ['Social Media', 'Search', 'Email', 'Mixed', 'Social Media']
        }
        df_campaigns = pd.DataFrame(campaign_data)
        
        # Customer data - simulate this based on orders if available
        if 'customer_id' in df_orders.columns:
            unique_customers = df_orders['customer_id'].unique()
            customer_data = {
                'customer_id': unique_customers,
                'signup_date': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(len(unique_customers))],
                'total_orders': [df_orders[df_orders['customer_id'] == cid].shape[0] for cid in unique_customers],
                'total_spent': [df_orders[df_orders['customer_id'] == cid]['amount'].sum() if 'amount' in df_orders.columns else np.random.randint(50, 5000) for cid in unique_customers],
                'segment': np.random.choice(['High Value', 'Regular', 'Occasional', 'New'], size=len(unique_customers))
            }
            df_customers = pd.DataFrame(customer_data)
        else:
            # Create sample customer data
            customer_data = {
                'customer_id': [f'CUST{i:04d}' for i in range(1, 501)],
                'signup_date': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(500)],
                'total_orders': np.random.randint(1, 20, size=500),
                'total_spent': np.random.randint(50, 5000, size=500),
                'segment': np.random.choice(['High Value', 'Regular', 'Occasional', 'New'], size=500)
            }
            df_customers = pd.DataFrame(customer_data)
            
        # Product data
        if 'product_id' in df_orders.columns:
            # Use actual product data from orders
            product_ids = df_orders['product_id'].unique()
            product_data = {
                'product_id': product_ids,
                'product_name': [f'Plant {i}' for i in product_ids],
                'category': np.random.choice(['Indoor', 'Outdoor', 'Succulents', 'Flowers', 'Tools'], size=len(product_ids)),
                'price': np.random.uniform(10, 200, size=len(product_ids)),
                'cost': np.random.uniform(5, 150, size=len(product_ids)),
                'stock': np.random.randint(0, 500, size=len(product_ids))
            }
            df_products = pd.DataFrame(product_data)
        else:
            # Create sample product data
            product_data = {
                'product_id': [f'P{i:03d}' for i in range(1, 101)],
                'product_name': [f'Plant {i}' for i in range(1, 101)],
                'category': np.random.choice(['Indoor', 'Outdoor', 'Succulents', 'Flowers', 'Tools'], size=100),
                'price': np.random.uniform(10, 200, size=100),
                'cost': np.random.uniform(5, 150, size=100),
                'stock': np.random.randint(0, 500, size=100)
            }
            df_products = pd.DataFrame(product_data)
            
        # If orders doesn't have expected columns, create a sample orders dataset
        if 'order_id' not in df_orders.columns:
            order_data = {
                'order_id': [f'ORD{i:06d}' for i in range(1, 10001)],
                'customer_id': np.random.choice(df_customers['customer_id'], size=10000),
                'order_date': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(10000)],
                'product_id': np.random.choice(df_products['product_id'], size=10000),
                'quantity': np.random.randint(1, 10, size=10000),
                'amount': np.random.uniform(10, 500, size=10000),
                'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Cash on Delivery'], size=10000),
                'status': np.random.choice(['Completed', 'Pending', 'Failed', 'Refunded'], 
                                         p=[0.85, 0.10, 0.03, 0.02], size=10000),
                'campaign_id': np.random.choice(df_campaigns['campaign_id'], size=10000)
            }
            df_orders = pd.DataFrame(order_data)
            
        # Financial data
        # Calculate based on orders if possible
        # Removed unused variable 'order_dates'
        daily_revenues = []
        daily_expenses = []
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        for date in dates:
            if 'order_date' in df_orders.columns and 'amount' in df_orders.columns:
                daily_rev = df_orders[pd.to_datetime(df_orders['order_date']).dt.date == date.date()]['amount'].sum()
            else:
                # Random revenue between 1000 and 5000
                daily_rev = np.random.uniform(1000, 5000)
            
            # Expenses are approximately 70% of revenue with some randomness
            daily_exp = daily_rev * np.random.uniform(0.6, 0.8)
            
            daily_revenues.append(daily_rev)
            daily_expenses.append(daily_exp)
            
        financial_data = {
            'date': dates,
            'revenue': daily_revenues,
            'expenses': daily_expenses,
            'profit': [rev - exp for rev, exp in zip(daily_revenues, daily_expenses)]
        }
        df_financial = pd.DataFrame(financial_data)
        
        return df_orders, df_campaigns, df_customers, df_products, df_financial
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrames in case of error
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Function to load models
@st.cache_resource
def load_models():
    try:
        with open("model.pkl", "rb") as file:
            models = pickle.load(file)
        return models
    except FileNotFoundError:
        # Train simple models if file not found
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=50)
        
        # Create a dictionary to store models
        models = {
            'linear_regression': linear_model,
            'random_forest': rf_model
        }
        
        # Save the models for future use
        with open("model.pkl", "wb") as file:
            pickle.dump(models, file)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Return empty dict in case of error
        return {}

# Load data and models
df_orders, df_campaigns, df_customers, df_products, df_financial = load_data()
models = load_models()

# Sidebar navigation
st.sidebar.markdown("<h1 style='text-align: center;'>🌱 Podha Plants</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to",
    ["Dashboard Overview", 
     "Marketing Campaign Analysis", 
     "Product & Customer Analysis", 
     "Financial Analysis", 
     "Strategic Planning",
     "Data Documentation"]
)

# Dashboard Overview
if page == "Dashboard Overview":
    st.markdown("<h1 class='main-header'>Podha Plants Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        This dashboard provides comprehensive analytics for Podha Plants, covering:
        <ul style='display: inline-block; text-align: left;'>
            <li>Marketing campaign performance</li>
            <li>Product and customer segmentation</li>
            <li>Financial analysis and risk management</li>
            <li>Strategic planning and growth opportunities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics in a 2x2 grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Orders", f"{len(df_orders):,}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        total_revenue = df_orders['amount'].sum() if 'amount' in df_orders.columns else df_financial['revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        customer_count = len(df_customers)
        st.metric("Total Customers", f"{customer_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        product_count = len(df_products)
        st.metric("Product Varieties", f"{product_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show recent trend
    st.markdown("<h2 class='section-header'>Recent Revenue Trend</h2>", unsafe_allow_html=True)
    
    # Create a revenue trend plot
    fig = px.line(
        df_financial.tail(30), 
        x='date', 
        y=['revenue', 'expenses', 'profit'],
        title='Last 30 Days Financial Performance',
        labels={'value': 'Amount ($)', 'date': 'Date', 'variable': 'Metric'},
        color_discrete_sequence=['#4CAF50', '#F44336', '#2196F3']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display quick insights
    st.markdown("<h2 class='section-header'>Quick Insights</h2>", unsafe_allow_html=True)
    
    # Insights based on data
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performing campaign
        if not df_campaigns.empty:
            best_campaign = df_campaigns.loc[df_campaigns['revenue'].idxmax()]
            st.info(f"**Best Performing Campaign:** {best_campaign['campaign_name']} with ${best_campaign['revenue']:,.2f} revenue")
        
        # Risky payment methods (if status exists)
        if 'status' in df_orders.columns and 'payment_method' in df_orders.columns:
            failed_payments = df_orders[df_orders['status'] == 'Failed'].groupby('payment_method').size()
            if not failed_payments.empty:
                riskiest_payment = failed_payments.idxmax()
                st.warning(f"**Highest Failure Rate:** {riskiest_payment} payment method")
    
    with col2:
        # Best selling product
        if 'product_id' in df_orders.columns:
            product_sales = df_orders.groupby('product_id').size()
            best_product_id = product_sales.idxmax()
            best_product = df_products[df_products['product_id'] == best_product_id]['product_name'].values[0] \
                           if best_product_id in df_products['product_id'].values else best_product_id
            st.success(f"**Best Selling Product:** {best_product} with {product_sales.max()} units sold")
        
        # Most valuable customer segment
        if 'segment' in df_customers.columns and 'total_spent' in df_customers.columns:
            segment_value = df_customers.groupby('segment')['total_spent'].sum()
            best_segment = segment_value.idxmax()
            st.info(f"**Most Valuable Segment:** {best_segment} with ${segment_value.max():,.2f} total spending")
            
# Marketing Campaign Analysis
elif page == "Marketing Campaign Analysis":
    st.markdown("<h1 class='main-header'>Marketing Campaign Analysis</h1>", unsafe_allow_html=True)
    
    # Campaign Overview
    st.markdown("<h2 class='section-header'>Campaign Overview</h2>", unsafe_allow_html=True)
    
    if not df_campaigns.empty:
        # Calculate KPIs
        df_campaigns['CAC'] = df_campaigns['cost'] / df_campaigns['conversions']
        df_campaigns['CTR'] = df_campaigns['clicks'] / df_campaigns['users_reached'] * 100
        df_campaigns['Conversion Rate'] = df_campaigns['conversions'] / df_campaigns['clicks'] * 100
        df_campaigns['ROI'] = (df_campaigns['revenue'] - df_campaigns['cost']) / df_campaigns['cost'] * 100
        df_campaigns['Profit'] = df_campaigns['revenue'] - df_campaigns['cost']
        
        # Display campaign metrics
        st.dataframe(df_campaigns[['campaign_name', 'cost', 'users_reached', 'clicks', 'conversions', 
                                 'revenue', 'CAC', 'CTR', 'Conversion Rate', 'ROI']])
        
        # Campaign Performance Comparison
        st.markdown("<h2 class='section-header'>Campaign Performance Comparison</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI Comparison
            fig_roi = px.bar(
                df_campaigns,
                x='campaign_name',
                y='ROI',
                title='Campaign ROI Comparison',
                color='ROI',
                text_auto='.1f',
                color_continuous_scale='RdYlGn'
            )
            fig_roi.update_layout(yaxis_title='ROI (%)')
            st.plotly_chart(fig_roi, use_container_width=True)
            
        with col2:
            # Profit Comparison
            fig_profit = px.bar(
                df_campaigns,
                x='campaign_name',
                y='Profit',
                title='Campaign Profit Comparison',
                color='Profit',
                text_auto='.2f',
                color_continuous_scale='Blues'
            )
            fig_profit.update_layout(yaxis_title='Profit ($)')
            st.plotly_chart(fig_profit, use_container_width=True)
        
        # Conversion Funnel
        st.markdown("<h2 class='section-header'>Conversion Funnel</h2>", unsafe_allow_html=True)
        
        # Let user select a campaign to view
        selected_campaign = st.selectbox(
            "Select Campaign for Detailed Analysis",
            df_campaigns['campaign_name'].tolist()
        )
        
        campaign = df_campaigns[df_campaigns['campaign_name'] == selected_campaign].iloc[0]
        
        # Create funnel chart
        funnel_data = dict(
            users_reached=campaign['users_reached'],
            clicks=campaign['clicks'],
            conversions=campaign['conversions']
        )
        
        funnel_stage_names = ["Users Reached", "Clicks", "Conversions"]
        funnel_values = [funnel_data['users_reached'], funnel_data['clicks'], funnel_data['conversions']]
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stage_names,
            x=funnel_values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=["#4CAF50", "#2196F3", "#9C27B0"])
        ))
        
        fig_funnel.update_layout(title=f"Conversion Funnel for {selected_campaign}")
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Campaign ROI Prediction
        st.markdown("<h2 class='section-header'>Campaign Budget Allocation Prediction</h2>", unsafe_allow_html=True)
        
        # Simple model for predicting conversions based on budget
        if 'linear_regression' in models:
            # Train a simple model using campaign data
            X = df_campaigns[['cost']].values
            y = df_campaigns['conversions'].values
            
            # Fit the model
            models['linear_regression'].fit(X, y)
            
            # User input for budget allocation
            budget = st.slider("Select Budget for Prediction ($)", 1000, 20000, 5000, 500)
            
            # Make prediction
            predicted_conversions = models['linear_regression'].predict([[budget]])[0]
            
            # Use the average conversion value to estimate revenue
            avg_conversion_value = df_campaigns['revenue'].sum() / df_campaigns['conversions'].sum()
            predicted_revenue = predicted_conversions * avg_conversion_value
            predicted_profit = predicted_revenue - budget
            predicted_roi = (predicted_profit / budget) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Conversions", f"{predicted_conversions:.0f}")
            
            with col2:
                st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
                
            with col3:
                st.metric("Predicted ROI", f"{predicted_roi:.1f}%")
                
            # Generate budget vs. conversions curve
            test_budgets = np.linspace(1000, 20000, 20)
            predicted_results = []
            
            for b in test_budgets:
                conv = models['linear_regression'].predict([[b]])[0]
                rev = conv * avg_conversion_value
                prof = rev - b
                roi = (prof / b) * 100
                predicted_results.append({
                    'Budget': b,
                    'Conversions': conv,
                    'Revenue': rev,
                    'Profit': prof,
                    'ROI': roi
                })
                
            df_predictions = pd.DataFrame(predicted_results)
            
            fig = px.line(
                df_predictions,
                x='Budget',
                y=['Conversions', 'Profit', 'ROI'],
                title='Budget Allocation Impact Prediction',
                labels={'value': 'Value', 'Budget': 'Budget ($)', 'variable': 'Metric'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("This prediction model uses a simple linear relationship between budget and conversions. In a real-world scenario, we would use more sophisticated models that account for diminishing returns and other factors.")
        
        # Acquisition Source Analysis
        st.markdown("<h2 class='section-header'>Acquisition Source Analysis</h2>", unsafe_allow_html=True)
        
        if 'acquisition_source' in df_campaigns.columns:
            source_performance = df_campaigns.groupby('acquisition_source').agg({
                'cost': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            source_performance['CAC'] = source_performance['cost'] / source_performance['conversions']
            source_performance['ROI'] = (source_performance['revenue'] - source_performance['cost']) / source_performance['cost'] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_source_cac = px.bar(
                    source_performance,
                    x='acquisition_source',
                    y='CAC',
                    title='Customer Acquisition Cost by Source',
                    color='acquisition_source'
                )
                fig_source_cac.update_layout(xaxis_title='Acquisition Source', yaxis_title='CAC ($)')
                st.plotly_chart(fig_source_cac, use_container_width=True)
                
            with col2:
                fig_source_roi = px.bar(
                    source_performance,
                    x='acquisition_source',
                    y='ROI',
                    title='ROI by Acquisition Source',
                    color='acquisition_source'
                )
                fig_source_roi.update_layout(xaxis_title='Acquisition Source', yaxis_title='ROI (%)')
                st.plotly_chart(fig_source_roi, use_container_width=True)
                
            st.markdown("**Recommendations based on Acquisition Source Performance:**")
            
            best_roi_source = source_performance.loc[source_performance['ROI'].idxmax()]['acquisition_source']
            lowest_cac_source = source_performance.loc[source_performance['CAC'].idxmin()]['acquisition_source']
            
            st.success(f"- Focus budget allocation on {best_roi_source} for highest ROI")
            st.success(f"- {lowest_cac_source} offers the lowest customer acquisition cost")
            
            if best_roi_source != lowest_cac_source:
                st.info(f"- Consider a balanced approach between {best_roi_source} and {lowest_cac_source}")
    else:
        st.warning("No campaign data available. Please upload campaign data to see analysis.")

# Product & Customer Analysis
elif page == "Product & Customer Analysis":
    st.markdown("<h1 class='main-header'>Product & Customer Segmentation Analysis</h1>", unsafe_allow_html=True)
    
    # Create tabs for Product Analysis and Customer Analysis
    tab1, tab2 = st.tabs(["Product Analysis", "Customer Segmentation"])
    
    with tab1:
        st.markdown("<h2 class='section-header'>Best-Selling Products</h2>", unsafe_allow_html=True)
        
        if 'product_id' in df_orders.columns:
            # Calculate product sales metrics
            product_sales = df_orders.groupby('product_id').agg({
                'order_id': 'count',
                'quantity': 'sum',
                'amount': 'sum' if 'amount' in df_orders.columns else None
            }).reset_index()
            
            product_sales.columns = ['product_id', 'orders', 'units_sold', 'revenue']
            
            # Merge with product data to get names
            if not df_products.empty:
                product_sales = product_sales.merge(df_products[['product_id', 'product_name', 'category', 'price', 'cost']], on='product_id', how='left')
                product_sales['profit'] = product_sales['revenue'] - (product_sales['units_sold'] * product_sales['cost'])
                product_sales['profit_margin'] = product_sales['profit'] / product_sales['revenue'] * 100
            
            # Sort by revenue
            product_sales = product_sales.sort_values('revenue', ascending=False)
            
            # Display top 10 products
            st.dataframe(product_sales.head(10))
            
            # Visualization of top products
            col1, col2 = st.columns(2)
            
            with col1:
                fig_top_revenue = px.bar(
                    product_sales.head(10),
                    x='product_name' if 'product_name' in product_sales.columns else 'product_id',
                    y='revenue',
                    title='Top 10 Products by Revenue',
                    color='category' if 'category' in product_sales.columns else None
                )
                st.plotly_chart(fig_top_revenue, use_container_width=True)
                
            with col2:
                fig_top_units = px.bar(
                    product_sales.head(10),
                    x='product_name' if 'product_name' in product_sales.columns else 'product_id',
                    y='units_sold',
                    title='Top 10 Products by Units Sold',
                    color='category' if 'category' in product_sales.columns else None
                )
                st.plotly_chart(fig_top_units, use_container_width=True)
            
            # Category performance
            if 'category' in product_sales.columns:
                st.markdown("<h2 class='section-header'>Category Performance</h2>", unsafe_allow_html=True)
                
                category_performance = product_sales.groupby('category').agg({
                    'orders': 'sum',
                    'units_sold': 'sum',
                    'revenue': 'sum',
                    'profit': 'sum'
                }).reset_index()
                
                category_performance['profit_margin'] = category_performance['profit'] / category_performance['revenue'] * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_category_revenue = px.pie(
                        category_performance,
                        values='revenue',
                        names='category',
                        title='Revenue Distribution by Category',
                        hole=0.4
                    )
                    st.plotly_chart(fig_category_revenue, use_container_width=True)
                    
                with col2:
                    fig_category_profit = px.bar(
                        category_performance,
                        x='category',
                        y=['revenue', 'profit'],
                        title='Revenue vs. Profit by Category',
                        barmode='group'
                    )
                    st.plotly_chart(fig_category_profit, use_container_width=True)
            
            # Trending Products Analysis
            st.markdown("<h2 class='section-header'>Trending Products Analysis</h2>", unsafe_allow_html=True)
            
            if 'order_date' in df_orders.columns:
                # Convert order_date to datetime if not already
                df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
                
                # Get the last 30 days of data
                last_30_days = df_orders[df_orders['order_date'] >= df_orders['order_date'].max() - pd.Timedelta(days=30)]
                
                # Calculate recent sales
                recent_sales = last_30_days.groupby('product_id').agg({
                    'order_id': 'count',
                    'quantity': 'sum',
                    'amount': 'sum' if 'amount' in last_30_days.columns else None
                }).reset_index()
                
                recent_sales.columns = ['product_id', 'recent_orders', 'recent_units', 'recent_revenue']
                
                # Merge with product data
                if not df_products.empty:
                    recent_sales = recent_sales.merge(df_products[['product_id', 'product_name', 'category']], on='product_id', how='left')
                
                # Calculate growth (if we have historical data)
                previous_30_days = df_orders[(df_orders['order_date'] < df_orders['order_date'].max() - pd.Timedelta(days=30)) & 
                                            (df_orders['order_date'] >= df_orders['order_date'].max() - pd.Timedelta(days=60))]
                
                if not previous_30_days.empty:
                    previous_sales = previous_30_days.groupby('product_id').agg({
                        'quantity': 'sum'
                    }).reset_index()
                    
                    previous_sales.columns = ['product_id', 'previous_units']
                    
                    # Merge with recent sales
                    sales_comparison = recent_sales.merge(previous_sales, on='product_id', how='left')
                    sales_comparison['previous_units'] = sales_comparison['previous_units'].fillna(0)
                    
                    # Calculate growth rate
                    sales_comparison['growth_rate'] = ((sales_comparison['recent_units'] - sales_comparison['previous_units']) / 
                                                     (sales_comparison['previous_units'] + 1)) * 100  # Add 1 to avoid division by zero
                    
                    # Sort by growth rate
                    trending_products = sales_comparison.sort_values('growth_rate', ascending=False)
                    
                    # Display trending products
                    st.dataframe(trending_products.head(10)[['product_name', 'category', 'recent_units', 'previous_units', 'growth_rate']])
                    
                    # Visualization of trending products
                    fig_trending = px.bar(
                        trending_products.head(10),
                        x='product_name',
                        y='growth_rate',
                        title='Top 10 Trending Products (Growth Rate)',
                        color='growth_rate',
                        text_auto='.1f',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_trending, use_container_width=True)
                else:
                    # If we don't have historical data, just show recent sales
                    trending_products = recent_sales.sort_values('recent_units', ascending=False)
                    st.dataframe(trending_products.head(10))
                    
                    fig_recent = px.bar(
                        trending_products.head(10),
                        x='product_name' if 'product_name' in trending_products.columns else 'product_id',
                        y='recent_units',
                        title='Top 10 Products (Last 30 Days)',
                        color='category' if 'category' in trending_products.columns else None
                    )
                    st.plotly_chart(fig_recent, use_container_width=True)
        else:
            st.warning("No product order data available. Please upload order data with product information to see analysis.")
    
    with tab2:
        st.markdown("<h2 class='section-header'>Customer Segmentation Analysis</h2>", unsafe_allow_html=True)
        
        if not df_customers.empty:
            # Display customer metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Customers", f"{len(df_customers):,}")
                
            with col2:
                avg_orders = df_customers['total_orders'].mean() if 'total_orders' in df_customers.columns else 0
                st.metric("Average Orders per Customer", f"{avg_orders:.2f}")
                
            with col3:
                avg_spent = df_customers['total_spent'].mean() if 'total_spent' in df_customers.columns else 0
                st.metric("Average Lifetime Value", f"${avg_spent:.2f}")
            
            # Customer segmentation visualization
            if 'segment' in df_customers.columns:
                st.markdown("<h3>Customer Segments</h3>", unsafe_allow_html=True)
                
                segment_counts = df_customers['segment'].value_counts().reset_index()
                segment_counts.columns = ['segment', 'count']
                
                segment_value = df_customers.groupby('segment').agg({
                    'total_spent': 'sum',
                    'total_orders': 'sum'
                }).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_segment_count = px.pie(
                        segment_counts,
                        values='count',
                        names='segment',
                        title='Customer Distribution by Segment',
                        hole=0.4
                    )
                    st.plotly_chart(fig_segment_count, use_container_width=True)
                    
                with col2:
                    fig_segment_value = px.bar(
                        segment_value,
                        x='segment',
                        y='total_spent',
                        title='Total Value by Customer Segment',
                        color='segment'
                    )
                    st.plotly_chart(fig_segment_value, use_container_width=True)
            
            # RFM Analysis (Recency, Frequency, Monetary)
            st.markdown("<h2 class='section-header'>RFM Segmentation</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            RFM Analysis segments customers based on three metrics:
            - **Recency** - How recently did the customer make a purchase?
            - **Frequency** - How often does the customer make purchases?
            - **Monetary** - How much does the customer spend?
            """)
            
            # Perform simple RFM analysis if we have the necessary data
            if 'order_date' in df_orders.columns and 'customer_id' in df_orders.columns and 'amount' in df_orders.columns:
                # Convert order_date to datetime if not already
                df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
                
                # Calculate RFM metrics
                today = df_orders['order_date'].max()
                
                rfm = df_orders.groupby('customer_id').agg({
                    'order_date': lambda x: (today - x.max()).days,  # Recency
                    'order_id': 'count',  # Frequency
                    'amount': 'sum'  # Monetary
                }).reset_index()
                
                rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
                
                # Create RFM scores
                rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])  # Smaller recency is better
                rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
                rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
                
                # Calculate RFM Score
                rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
                
                # Create segment based on RFM score
                def segment_customer(row):
                    if row['R_score'] == 5 and row['F_score'] == 5 and row['M_score'] == 5:
                        return 'Champions'
                    elif row['R_score'] >= 3 and row['F_score'] >= 3 and row['M_score'] >= 3:
                        return 'Loyal Customers'
                    elif row['R_score'] >= 4 and row['F_score'] >= 1 and row['M_score'] >= 4:
                        return 'Potential Loyalists'
                    elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] <= 2:
                        return 'At Risk'
                    elif row['R_score'] == 1:
                        return 'Lost Customers'
                    elif row['F_score'] == 5 and row['M_score'] >= 4:
                        return "Can't Lose Them"
                    else:
                        return 'Others'
                
                rfm['RFM_segment'] = rfm.apply(segment_customer, axis=1)
                
                # Display RFM insights
                st.markdown("<h3>RFM Segment Distribution</h3>", unsafe_allow_html=True)
                
                segment_counts = rfm['RFM_segment'].value_counts().reset_index()
                segment_counts.columns = ['segment', 'count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rfm_segments = px.pie(
                        segment_counts,
                        values='count',
                        names='segment',
                        title='Customer Distribution by RFM Segment',
                        hole=0.4
                    )
                    st.plotly_chart(fig_rfm_segments, use_container_width=True)
                
                with col2:
                    segment_value = rfm.groupby('RFM_segment')['monetary'].sum().reset_index()
                    segment_value.columns = ['segment', 'total_value']
                    
                    fig_segment_value = px.bar(
                        segment_value,
                        x='segment',
                        y='total_value',
                        title='Total Value by RFM Segment',
                        color='segment'
                    )
                    st.plotly_chart(fig_segment_value, use_container_width=True)
                
                # Segment-specific insights
                st.markdown("<h3>RFM Segment Insights</h3>", unsafe_allow_html=True)
                
                segment_metrics = rfm.groupby('RFM_segment').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'mean',
                    'customer_id': 'count'
                }).reset_index()
                
                segment_metrics.columns = ['segment', 'avg_recency', 'avg_frequency', 'avg_monetary', 'count']
                segment_metrics['avg_recency'] = segment_metrics['avg_recency'].round(1)
                segment_metrics['avg_frequency'] = segment_metrics['avg_frequency'].round(1)
                segment_metrics['avg_monetary'] = segment_metrics['avg_monetary'].round(2)
                
                st.dataframe(segment_metrics)
                
                # Customer segment product preferences
                st.markdown("<h3>Product Preferences by Customer Segment</h3>", unsafe_allow_html=True)
                
                if 'product_id' in df_orders.columns:
                    # Merge orders with RFM segments
                    orders_with_segment = df_orders.merge(rfm[['customer_id', 'RFM_segment']], on='customer_id', how='left')
                    
                    # Get product preferences by segment
                    segment_products = orders_with_segment.groupby(['RFM_segment', 'product_id']).agg({
                        'order_id': 'count'
                    }).reset_index()
                    
                    segment_products.columns = ['segment', 'product_id', 'purchase_count']
                    
                    # Get top product for each segment
                    top_products = segment_products.sort_values('purchase_count', ascending=False).groupby('segment').head(1)
                    
                    # Merge with product data to get names
                    if not df_products.empty and 'product_name' in df_products.columns:
                        top_products = top_products.merge(df_products[['product_id', 'product_name']], on='product_id', how='left')
                        
                        # Display top product for each segment
                        st.dataframe(top_products[['segment', 'product_name', 'purchase_count']])
                    else:
                        st.dataframe(top_products)
            else:
                st.info("Complete order history with customer IDs and dates is needed for RFM analysis.")
                
            # Customer Clustering using K-means
            st.markdown("<h2 class='section-header'>Advanced Customer Clustering</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            K-means clustering can identify natural groupings in customer behavior beyond pre-defined segments.
            This can reveal insights that traditional segmentation might miss.
            """)
            
            if 'total_orders' in df_customers.columns and 'total_spent' in df_customers.columns:
                # Select features for clustering
                features = ['total_orders', 'total_spent']
                X = df_customers[features].values
                
                # Standardize the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Determine optimal number of clusters
                wcss = []
                max_clusters = min(10, len(df_customers) // 10)  # Don't try too many clusters for small datasets
                
                for i in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)
                
                # Plot elbow method chart
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=list(range(1, max_clusters + 1)), y=wcss, mode='lines+markers'))
                fig_elbow.update_layout(title='Elbow Method for Optimal k',
                                       xaxis_title='Number of Clusters',
                                       yaxis_title='WCSS')
                
                st.plotly_chart(fig_elbow, use_container_width=True)
                
                # Let user select number of clusters
                n_clusters = st.slider("Select Number of Clusters", 2, max_clusters, 4)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
                df_customers['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # Visualize clusters
                fig_clusters = px.scatter(
                    df_customers,
                    x='total_orders',
                    y='total_spent',
                    color='Cluster',
                    title=f'Customer Clusters (k={n_clusters})'
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
                
                # Show cluster characteristics
                cluster_stats = df_customers.groupby('Cluster').agg({
                    'customer_id': 'count',
                    'total_orders': 'mean',
                    'total_spent': 'mean'
                }).reset_index()
                
                cluster_stats.columns = ['Cluster', 'Customer Count', 'Avg Orders', 'Avg Spent']
                cluster_stats['Avg Orders'] = cluster_stats['Avg Orders'].round(1)
                cluster_stats['Avg Spent'] = cluster_stats['Avg Spent'].round(2)
                
                st.dataframe(cluster_stats)
                
                # Cluster insights
                st.markdown("<h3>Cluster Insights</h3>", unsafe_allow_html=True)
                
                # Find highest value cluster
                high_value_cluster = cluster_stats.loc[cluster_stats['Avg Spent'].idxmax()]['Cluster']
                high_order_cluster = cluster_stats.loc[cluster_stats['Avg Orders'].idxmax()]['Cluster']
                
                st.success(f"Cluster {high_value_cluster} represents your highest value customers with average spend of ${cluster_stats.loc[cluster_stats['Cluster'] == high_value_cluster, 'Avg Spent'].values[0]:.2f}")
                
                if high_order_cluster != high_value_cluster:
                    st.info(f"Cluster {high_order_cluster} represents your most frequent buyers with average {cluster_stats.loc[cluster_stats['Cluster'] == high_order_cluster, 'Avg Orders'].values[0]:.1f} orders")
                
                # Marketing recommendations for each cluster
                st.markdown("<h3>Cluster Marketing Recommendations</h3>", unsafe_allow_html=True)
                
                for cluster in sorted(df_customers['Cluster'].unique()):
                    cluster_avg_spend = cluster_stats.loc[cluster_stats['Cluster'] == cluster, 'Avg Spent'].values[0]
                    cluster_avg_orders = cluster_stats.loc[cluster_stats['Cluster'] == cluster, 'Avg Orders'].values[0]
                    
                    st.markdown(f"**Cluster {cluster}**")
                    
                    if cluster == high_value_cluster:
                        st.markdown("- Focus on retention through premium loyalty programs")
                        st.markdown("- Offer early access to new products")
                        st.markdown("- Create exclusive VIP experiences")
                    elif cluster == high_order_cluster:
                        st.markdown("- Encourage larger basket sizes with bundle discounts")
                        st.markdown("- Implement a tiered loyalty program")
                        st.markdown("- Focus on cross-selling complementary products")
                    elif cluster_avg_spend > cluster_stats['Avg Spent'].median():
                        st.markdown("- Increase purchase frequency with personalized offers")
                        st.markdown("- Focus on category expansion to increase spend")
                    else:
                        st.markdown("- Focus on activation and engagement")
                        st.markdown("- Offer entry-level products with good margins")
                        st.markdown("- Use limited-time offers to encourage trial")
                    
                    st.markdown("")
        else:
            st.warning("No customer data available. Please upload customer data to see segmentation analysis.")

# Financial Analysis
elif page == "Financial Analysis":
    st.markdown("<h1 class='main-header'>Financial Analysis</h1>", unsafe_allow_html=True)
    
    # Create tabs for different financial analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Financial Performance", "Risk Management", "Transaction Analysis", "Revenue Breakdown"])
    
    with tab1:
        st.markdown("<h2 class='section-header'>Financial Performance</h2>", unsafe_allow_html=True)
        
        if not df_financial.empty:
            # Calculate key financial metrics
            total_revenue = df_financial['revenue'].sum()
            total_expenses = df_financial['expenses'].sum()
            total_profit = df_financial['profit'].sum()
            profit_margin = (total_profit / total_revenue) * 100
            
            # Display financial metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
                
            with col2:
                st.metric("Total Expenses", f"${total_expenses:,.2f}")
                
            with col3:
                st.metric("Total Profit", f"${total_profit:,.2f}")
                
            with col4:
                st.metric("Profit Margin", f"{profit_margin:.2f}%")
            
            # Time period selection
            time_period = st.selectbox(
                "Select Time Period",
                ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"]
            )
            
            # Filter data based on time period
            if time_period == "All Time":
                filtered_financials = df_financial
            elif time_period == "Last 30 Days":
                filtered_financials = df_financial[df_financial['date'] >= df_financial['date'].max() - pd.Timedelta(days=30)]
            elif time_period == "Last 90 Days":
                filtered_financials = df_financial[df_financial['date'] >= df_financial['date'].max() - pd.Timedelta(days=90)]
            elif time_period == "Last 6 Months":
                filtered_financials = df_financial[df_financial['date'] >= df_financial['date'].max() - pd.Timedelta(days=180)]
            elif time_period == "Last Year":
                filtered_financials = df_financial[df_financial['date'] >= df_financial['date'].max() - pd.Timedelta(days=365)]
            
            # Monthly financial trends
            monthly_financials = filtered_financials.copy()
            monthly_financials['month'] = monthly_financials['date'].dt.to_period('M')
            monthly_financials = monthly_financials.groupby('month').agg({
                'revenue': 'sum',
                'expenses': 'sum',
                'profit': 'sum'
            }).reset_index()
            
            monthly_financials['month'] = monthly_financials['month'].astype(str)
            monthly_financials['profit_margin'] = (monthly_financials['profit'] / monthly_financials['revenue']) * 100
            
            # Create financial trend chart
            fig_financial = go.Figure()
            
            fig_financial.add_trace(go.Bar(
                x=monthly_financials['month'],
                y=monthly_financials['revenue'],
                name='Revenue',
                marker_color='#4CAF50'
            ))
            
            fig_financial.add_trace(go.Bar(
                x=monthly_financials['month'],
                y=monthly_financials['expenses'],
                name='Expenses',
                marker_color='#F44336'
            ))
            
            fig_financial.add_trace(go.Scatter(
                x=monthly_financials['month'],
                y=monthly_financials['profit'],
                name='Profit',
                mode='lines+markers',
                line=dict(color='#2196F3', width=3)
            ))
            
            fig_financial.update_layout(
                title='Monthly Financial Performance',
                barmode='group',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_financial, use_container_width=True)
            
            # Profit margin trend
            fig_margin = px.line(
                monthly_financials,
                x='month',
                y='profit_margin',
                title='Monthly Profit Margin Trend',
                markers=True
            )
            
            fig_margin.update_layout(
                xaxis_title='Month',
                yaxis_title='Profit Margin (%)'
            )
            
            st.plotly_chart(fig_margin, use_container_width=True)
            
            # Financial forecast using simple model
            st.markdown("<h2 class='section-header'>Financial Forecast</h2>", unsafe_allow_html=True)
            
            # Train a simple model using time series data
            if 'random_forest' in models:
                # Prepare data
                df_financial['day_of_year'] = df_financial['date'].dt.dayofyear
                df_financial['month'] = df_financial['date'].dt.month
                df_financial['day_of_week'] = df_financial['date'].dt.dayofweek
                
                X = df_financial[['day_of_year', 'month', 'day_of_week']].values
                y_revenue = df_financial['revenue'].values
                y_expenses = df_financial['expenses'].values
                
                # Fit models
                models['random_forest'].fit(X, y_revenue)
                
                # Create features for the next 30 days
                last_date = df_financial['date'].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
                
                forecast_X = pd.DataFrame({
                    'day_of_year': forecast_dates.dayofyear,
                    'month': forecast_dates.month,
                    'day_of_week': forecast_dates.dayofweek
                }).values
                
                # Make predictions
                forecast_revenue = models['random_forest'].predict(forecast_X)
                forecast_expenses = forecast_revenue * 0.7  # Simplified expense model
                forecast_profit = forecast_revenue - forecast_expenses
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'revenue': forecast_revenue,
                    'expenses': forecast_expenses,
                    'profit': forecast_profit
                })
                
                # Create forecast chart
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=df_financial['date'].tail(30),
                    y=df_financial['revenue'].tail(30),
                    name='Historical Revenue',
                    line=dict(color='#4CAF50', width=2, dash='dash')
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=df_financial['date'].tail(30),
                    y=df_financial['profit'].tail(30),
                    name='Historical Profit',
                    line=dict(color='#2196F3', width=2, dash='dash')
                ))
                
                # Forecast data
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['revenue'],
                    name='Forecast Revenue',
                    line=dict(color='#4CAF50', width=3)
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['profit'],
                    name='Forecast Profit',
                    line=dict(color='#2196F3', width=3)
                ))
                
                fig_forecast.update_layout(
                    title='30-Day Financial Forecast',
                    xaxis_title='Date',
                    yaxis_title='Amount ($)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Summary metrics
                forecast_total_revenue = forecast_df['revenue'].sum()
                forecast_total_profit = forecast_df['profit'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Forecasted 30-Day Revenue", f"${forecast_total_revenue:,.2f}")
                    
                with col2:
                    st.metric("Forecasted 30-Day Profit", f"${forecast_total_profit:,.2f}")
                
                st.info("This forecast uses a simple RandomForest model trained on historical data. In a real-world scenario, we would use more sophisticated models with additional features and seasonality adjustments.")
        else:
            st.warning("No financial data available. Please upload financial data to see analysis.")
    
    with tab2:
        st.markdown("<h2 class='section-header'>Risk Management</h2>", unsafe_allow_html=True)
        
        if 'status' in df_orders.columns:
            # Calculate payment failure metrics
            payment_status = df_orders['status'].value_counts().reset_index()
            payment_status.columns = ['status', 'count']
            
            total_transactions = payment_status['count'].sum()
            failed_transactions = payment_status[payment_status['status'] == 'Failed']['count'].sum() if 'Failed' in payment_status['status'].values else 0
            failed_rate = (failed_transactions / total_transactions) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Failed Transactions", f"{failed_transactions:,}")
                
            with col2:
                st.metric("Failure Rate", f"{failed_rate:.2f}%")
            
            # Payment status visualization
            fig_status = px.pie(
                payment_status,
                values='count',
                names='status',
                title='Transaction Status Distribution',
                hole=0.4,
                color_discrete_map={
                    'Completed': '#4CAF50',
                    'Pending': '#FFC107',
                    'Failed': '#F44336',
                    'Refunded': '#9C27B0'
                }
            )
            st.plotly_chart(fig_status, use_container_width=True)
            
            # Payment method risk analysis
            if 'payment_method' in df_orders.columns:
                payment_risk = df_orders.groupby(['payment_method', 'status']).size().reset_index()
                payment_risk.columns = ['payment_method', 'status', 'count']
                
                # Calculate failure rates for each payment method
                payment_totals = df_orders.groupby('payment_method').size().reset_index()
                payment_totals.columns = ['payment_method', 'total']
                
                payment_failures = payment_risk[payment_risk['status'] == 'Failed'] if 'Failed' in payment_risk['status'].values else pd.DataFrame(columns=['payment_method', 'status', 'count'])
                
                if not payment_failures.empty:
                    payment_failures = payment_failures.merge(payment_totals, on='payment_method', how='left')
                    payment_failures['failure_rate'] = (payment_failures['count'] / payment_failures['total']) * 100
                    
                    # Sort by failure rate
                    payment_failures = payment_failures.sort_values('failure_rate', ascending=False)
                    
                    # Display payment method risk
                    st.markdown("<h3>Payment Method Risk Analysis</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_method_risk = px.bar(
                            payment_failures,
                            x='payment_method',
                            y='failure_rate',
                            title='Payment Method Failure Rates',
                            color='failure_rate',
                            color_continuous_scale='Reds',
                            text_auto='.2f'
                        )
                        fig_method_risk.update_layout(yaxis_title='Failure Rate (%)')
                        st.plotly_chart(fig_method_risk, use_container_width=True)
                    
                    with col2:
                        payment_method_status = pd.crosstab(df_orders['payment_method'], df_orders['status'])
                        
                        fig_method_status = px.bar(
                            payment_method_status,
                            title='Transaction Status by Payment Method',
                            barmode='stack',
                            color_discrete_map={
                                'Completed': '#4CAF50',
                                'Pending': '#FFC107',
                                'Failed': '#F44336',
                                'Refunded': '#9C27B0'
                            }
                        )
                        st.plotly_chart(fig_method_status, use_container_width=True)
                    
                    # Risk mitigation recommendations
                    st.markdown("<h3>Risk Mitigation Recommendations</h3>", unsafe_allow_html=True)
                    
                    highest_risk_method = payment_failures.iloc[0]['payment_method']
                    highest_risk_rate = payment_failures.iloc[0]['failure_rate']
                    
                    st.warning(f"**Highest Risk Payment Method:** {highest_risk_method} with {highest_risk_rate:.2f}% failure rate")
                    
                    st.markdown("""
                    **Recommended Actions:**
                    1. Investigate the root causes of failures for high-risk payment methods
                    2. Implement additional verification steps for high-risk methods
                    3. Offer incentives for customers to use more reliable payment methods
                    4. Establish automated retry mechanisms for failed transactions
                    5. Create a dedicated monitoring system for payment failures
                    """)
                else:
                    st.info("No failed transactions found in the dataset.")
            else:
                st.info("Payment method data not available for detailed risk analysis.")
            
            # Financial impact of failed transactions
            if 'amount' in df_orders.columns:
                st.markdown("<h3>Financial Impact of Failed Transactions</h3>", unsafe_allow_html=True)
                
                status_financials = pd.DataFrame()  # Placeholder to avoid undefined variable error
