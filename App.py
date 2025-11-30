# app.py - Streamlit app to load and visualize the stock prediction model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stock-header {
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data():
    """Load the stock data from pickle file"""
    try:
        with open('stock_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ stock_data.pkl file not found. Please run generate_data.py first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### MADSC102 Final Project - Multi-Feature ML Prediction")
    
    # Load data
    with st.spinner("Loading stock data..."):
        stock_data = load_stock_data()
    
    if stock_data is None:
        st.stop()
    
    # Display data overview
    st.sidebar.markdown("## 📊 Data Overview")
    st.sidebar.write(f"**Total Records:** {len(stock_data):,}")
    st.sidebar.write(f"**Stocks:** {', '.join(stock_data['symbol'].unique())}")
    st.sidebar.write(f"**Date Range:** {stock_data['date'].min().strftime('%Y-%m-%d')} to {stock_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")
    
    # Stock selection
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks:",
        options=stock_data['symbol'].unique(),
        default=stock_data['symbol'].unique()[:3]
    )
    
    # Date range selection
    min_date = stock_data['date'].min()
    max_date = stock_data['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on selections
    if len(selected_stocks) > 0 and len(date_range) == 2:
        filtered_data = stock_data[
            (stock_data['symbol'].isin(selected_stocks)) &
            (stock_data['date'] >= pd.to_datetime(date_range[0])) &
            (stock_data['date'] <= pd.to_datetime(date_range[1]))
        ].copy()
    else:
        filtered_data = stock_data.copy()
    
    # Main dashboard
    if len(filtered_data) == 0:
        st.warning("No data available for the selected filters.")
        return
    
    # Key Metrics
    st.markdown("## 📋 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_close = filtered_data['close_price'].mean()
        st.metric("Average Close Price", f"${avg_close:.2f}")
    
    with col2:
        avg_volume = filtered_data['volume'].mean()
        st.metric("Average Volume", f"{avg_volume:,.0f}")
    
    with col3:
        avg_pred_error = filtered_data['prediction_error'].abs().mean()
        st.metric("Avg Prediction Error", f"${avg_pred_error:.2f}")
    
    with col4:
        total_records = len(filtered_data)
        st.metric("Total Records", f"{total_records:,}")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🔮 Predictions", "📊 Stock Comparison", "📋 Raw Data"])
    
    with tab1:
        st.markdown("### Stock Price Trends")
        
        # Price type selection
        price_type = st.selectbox(
            "Select Price Type:",
            ['open_price', 'high_price', 'low_price', 'close_price'],
            index=3
        )
        
        # Create interactive price chart
        fig = px.line(
            filtered_data,
            x='date',
            y=price_type,
            color='symbol',
            title=f'{price_type.replace("_", " ").title()} Over Time',
            labels={price_type: 'Price ($)', 'date': 'Date'}
        )
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Prediction Analysis")
        
        # Select a specific stock for detailed prediction analysis
        selected_stock = st.selectbox(
            "Select Stock for Prediction Analysis:",
            filtered_data['symbol'].unique()
        )
        
        stock_data_filtered = filtered_data[filtered_data['symbol'] == selected_stock]
        
        # Create prediction vs actual chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stock_data_filtered['date'],
            y=stock_data_filtered['close_price'],
            mode='lines',
            name='Actual Close Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=stock_data_filtered['date'],
            y=stock_data_filtered['prediction'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted Prices for {selected_stock}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction accuracy metrics
        st.markdown("#### Prediction Accuracy")
        pred_errors = stock_data_filtered['prediction_error']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            mae = pred_errors.abs().mean()
            st.metric("Mean Absolute Error", f"${mae:.2f}")
        
        with col2:
            rmse = np.sqrt((pred_errors ** 2).mean())
            st.metric("Root Mean Square Error", f"${rmse:.2f}")
        
        with col3:
            accuracy_within_1 = (pred_errors.abs() <= 1).mean() * 100
            st.metric("Accuracy within $1", f"{accuracy_within_1:.1f}%")
    
    with tab3:
        st.markdown("### Stock Comparison")
        
        # Comparison metrics
        comparison_metric = st.selectbox(
            "Select Metric for Comparison:",
            ['close_price', 'volume', 'prediction_error', 'high_price', 'low_price']
        )
        
        # Create comparison chart
        if comparison_metric == 'prediction_error':
            # For errors, show absolute values
            comp_data = filtered_data.groupby('symbol')[comparison_metric].apply(lambda x: x.abs().mean()).reset_index()
            y_label = 'Absolute Error ($)'
        else:
            comp_data = filtered_data.groupby('symbol')[comparison_metric].mean().reset_index()
            y_label = 'Average ' + comparison_metric.replace('_', ' ').title()
        
        fig = px.bar(
            comp_data,
            x='symbol',
            y=comparison_metric,
            title=f'{y_label} by Stock',
            labels={'symbol': 'Stock Symbol', comparison_metric: y_label}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Price Correlation Matrix")
        pivot_data = filtered_data.pivot_table(
            index='date', 
            columns='symbol', 
            values='close_price'
        ).corr()
        
        fig = px.imshow(
            pivot_data,
            title="Correlation Between Stock Prices",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Raw Data View")
        
        # Data summary
        st.write(f"Showing {len(filtered_data)} records")
        
        # Display data table
        st.dataframe(
            filtered_data.sort_values(['symbol', 'date']),
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**MADSC102 Final Project** | Built with Streamlit | "
        "Data: Generated Sample Stock Data"
    )

if __name__ == "__main__":
    main()
