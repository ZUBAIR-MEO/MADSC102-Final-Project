# app.py - Streamlit app to load and visualize the stock prediction model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using basic charts.")

@st.cache_data
def load_stock_data():
    """Load the stock data from pickle file"""
    try:
        with open('stock_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ stock_data.pkl file not found. Please run generate_data.py first.")
        st.info("""
        **To generate the data file:**
        1. Create a file called `generate_data.py` with the code provided
        2. Run: `python generate_data.py`
        3. Restart this app
        """)
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def create_basic_chart(data, x_col, y_col, color_col, title):
    """Create a basic line chart using Streamlit's native charting"""
    chart_data = data.pivot_table(
        index=x_col, 
        columns=color_col, 
        values=y_col
    ).reset_index()
    
    st.line_chart(chart_data.set_index(x_col), use_container_width=True)

def main():
    # Header
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("### MADSC102 Final Project - Multi-Feature ML Prediction")
    
    # Load data
    with st.spinner("Loading stock data..."):
        stock_data = load_stock_data()
    
    if stock_data is None:
        # Show instructions for first-time setup
        st.markdown("""
        ## First-time Setup Required
        
        1. **Create the data file** by running this Python code:
        ```python
        # generate_data.py
        import pandas as pd
        import numpy as np
        import pickle

        def generate_sample_data():
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', end='2024-12-10', freq='D')
            symbols = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
            
            all_data = []
            for symbol in symbols:
                base_price = np.random.uniform(100, 300)
                prices = [base_price]
                
                for i in range(1, len(dates)):
                    change = np.random.normal(0, 2)
                    new_price = prices[-1] + change
                    new_price = max(new_price, 10)
                    prices.append(new_price)
                
                for i, date in enumerate(dates):
                    price = prices[i]
                    volume = np.random.randint(1000000, 50000000)
                    prediction = price + np.random.normal(0, 3)
                    
                    all_data.append({
                        'date': date,
                        'symbol': symbol,
                        'open_price': price - np.random.uniform(0, 2),
                        'high_price': price + np.random.uniform(0, 3),
                        'low_price': price - np.random.uniform(1, 4),
                        'close_price': price,
                        'volume': volume,
                        'prediction': prediction,
                        'prediction_error': np.random.normal(0, 2)
                    })
            
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            return df

        # Generate and save data
        stock_data = generate_sample_data()
        with open('stock_data.pkl', 'wb') as f:
            pickle.dump(stock_data, f)
        print("Data generated successfully!")
        ```
        
        2. **Run the script** in your environment
        3. **Restart this app**
        """)
        return
    
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
        
        if PLOTLY_AVAILABLE:
            # Use Plotly if available
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
        else:
            # Use Streamlit's native chart
            st.line_chart(
                filtered_data.pivot_table(
                    index='date', 
                    columns='symbol', 
                    values=price_type
                ),
                use_container_width=True
            )
    
    with tab2:
        st.markdown("### Prediction Analysis")
        
        # Select a specific stock for detailed prediction analysis
        selected_stock = st.selectbox(
            "Select Stock for Prediction Analysis:",
            filtered_data['symbol'].unique()
        )
        
        stock_data_filtered = filtered_data[filtered_data['symbol'] == selected_stock]
        
        if PLOTLY_AVAILABLE:
            # Create prediction vs actual chart with Plotly
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
        else:
            # Use Streamlit's native chart
            chart_data = stock_data_filtered[['date', 'close_price', 'prediction']].set_index('date')
            st.line_chart(chart_data, use_container_width=True)
        
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
        
        # Show error distribution
        st.markdown("#### Prediction Error Distribution")
        st.bar_chart(pred_errors.value_counts().sort_index())
    
    with tab3:
        st.markdown("### Stock Comparison")
        
        # Comparison metrics
        comparison_metric = st.selectbox(
            "Select Metric for Comparison:",
            ['close_price', 'volume', 'prediction_error', 'high_price', 'low_price']
        )
        
        # Create comparison chart
        if comparison_metric == 'prediction_error':
            comp_data = filtered_data.groupby('symbol')[comparison_metric].apply(lambda x: x.abs().mean()).reset_index()
            y_label = 'Absolute Error ($)'
        else:
            comp_data = filtered_data.groupby('symbol')[comparison_metric].mean().reset_index()
            y_label = 'Average ' + comparison_metric.replace('_', ' ').title()
        
        st.bar_chart(comp_data.set_index('symbol'))
        
        # Show correlation matrix as a table
        st.markdown("#### Price Correlation Matrix")
        pivot_data = filtered_data.pivot_table(
            index='date', 
            columns='symbol', 
            values='close_price'
        ).corr()
        
        st.dataframe(pivot_data.style.background_gradient(cmap='RdBu_r'), use_container_width=True)
    
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
