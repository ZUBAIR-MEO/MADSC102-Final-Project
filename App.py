# Enhanced Stock Prediction with Multiple Features
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Advanced Stock Predictor", layout="wide")
st.title("📈 Advanced Stock Price Prediction Dashboard")
st.markdown("**MADSC102 Final Project - Multi-Feature ML Prediction**")

@st.cache_data
def load_data():
    try:
        with open('stock_data.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return generate_enhanced_data()

def generate_enhanced_data():
    """Generate enhanced stock data with multiple predictive features"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    symbols = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
    
    data = []
    for symbol in symbols:
        # Generate realistic price trends with seasonality
        base_trend = np.linspace(100, 300, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Monthly seasonality
        noise = np.random.normal(0, 5, len(dates))
        prices = base_trend + seasonal + noise
        
        # Generate multiple technical indicators
        for i, date in enumerate(dates):
            if i >= 10:  # Need enough history for indicators
                # Price features
                current_price = max(prices[i], 10)
                prev_price = prices[i-1]
                
                # Technical indicators
                sma_5 = np.mean(prices[i-5:i])  # 5-day simple moving average
                sma_10 = np.mean(prices[i-10:i])  # 10-day simple moving average
                sma_20 = np.mean(prices[i-20:i])  # 20-day simple moving average
                
                # Volatility measures
                high_price = current_price + np.random.uniform(1, 3)
                low_price = current_price - np.random.uniform(1, 3)
                daily_range = high_price - low_price
                
                # Volume features
                base_volume = np.random.randint(1000000, 50000000)
                volume_ma_5 = base_volume * np.random.uniform(0.8, 1.2)
                
                # Momentum indicators
                price_momentum = current_price - prices[i-5]  # 5-day momentum
                rsi_like = np.random.uniform(30, 70)  # Simulated RSI
                
                # Market sentiment features
                market_sentiment = np.random.normal(0, 1)
                volatility_index = np.random.uniform(10, 30)
                
                # Economic indicators (simulated)
                interest_rate = np.random.uniform(2, 5)
                inflation_rate = np.random.uniform(1, 4)
                
                # Create target variable (next day's price)
                if i < len(dates) - 1:
                    next_day_price = prices[i+1]
                else:
                    next_day_price = current_price + np.random.normal(0, 2)
                
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'close_price': current_price,
                    'open_price': prev_price + np.random.normal(0, 1),
                    'high_price': high_price,
                    'low_price': low_price,
                    'volume': base_volume,
                    'volume_ma_5': volume_ma_5,
                    
                    # Technical indicators
                    'sma_5': sma_5,
                    'sma_10': sma_10,
                    'sma_20': sma_20,
                    'daily_range': daily_range,
                    'price_momentum': price_momentum,
                    'rsi_like': rsi_like,
                    
                    # Market features
                    'market_sentiment': market_sentiment,
                    'volatility_index': volatility_index,
                    
                    # Economic indicators
                    'interest_rate': interest_rate,
                    'inflation_rate': inflation_rate,
                    
                    # Target variable
                    'next_day_price': next_day_price,
                    'price_change': next_day_price - current_price,
                    'price_change_pct': (next_day_price - current_price) / current_price * 100
                })
    
    df = pd.DataFrame(data)
    return df

# Load data
df = load_data()

# Sidebar for configuration
st.sidebar.header("🎯 Prediction Configuration")

# Stock selection
symbols = df['symbol'].unique()
selected_symbol = st.sidebar.selectbox("Select Stock", symbols)

# Model selection
model_type = st.sidebar.selectbox(
    "Select ML Model",
    ["Linear Regression", "Random Forest", "Both"]
)

# Feature selection
st.sidebar.subheader("Feature Selection")
feature_groups = {
    "Price Features": ['open_price', 'high_price', 'low_price', 'close_price'],
    "Technical Indicators": ['sma_5', 'sma_10', 'sma_20', 'daily_range', 'price_momentum', 'rsi_like'],
    "Volume Features": ['volume', 'volume_ma_5'],
    "Market Indicators": ['market_sentiment', 'volatility_index'],
    "Economic Factors": ['interest_rate', 'inflation_rate']
}

selected_features = []
for group, features in feature_groups.items():
    if st.sidebar.checkbox(group, value=True):
        selected_features.extend(features)

# Filter data for selected symbol
symbol_data = df[df['symbol'] == selected_symbol].copy()

if len(symbol_data) > 0 and len(selected_features) > 0:
    # Prepare features and target
    X = symbol_data[selected_features]
    y = symbol_data['next_day_price']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train models
    models = {}
    predictions = {}
    
    if model_type in ["Linear Regression", "Both"]:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        models['Linear Regression'] = lr_model
        predictions['Linear Regression'] = lr_model.predict(X_test)
    
    if model_type in ["Random Forest", "Both"]:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        predictions['Random Forest'] = rf_model.predict(X_test)
    
    # Calculate metrics
    metrics_data = []
    feature_importance_data = []
    
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics_data.append({
            'Model': model_name,
            'MAE': f"${mae:.2f}",
            'RMSE': f"${rmse:.2f}",
            'R² Score': f"{r2:.4f}",
            'Accuracy': f"{(1 - mae/y_test.mean()) * 100:.1f}%"
        })
        
        # Feature importance for Random Forest
        if model_name == "Random Forest":
            importance = models[model_name].feature_importances_
            for feature, imp in zip(selected_features, importance):
                feature_importance_data.append({
                    'Feature': feature,
                    'Importance': f"{imp:.4f}",
                    'Model': model_name
                })
    
    # Display results
    st.header(f"📊 {selected_symbol} Price Prediction Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = symbol_data['close_price'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        avg_volume = symbol_data['volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    with col3:
        volatility = symbol_data['daily_range'].mean()
        st.metric("Avg Volatility", f"${volatility:.2f}")
    with col4:
        if len(metrics_data) > 0:
            best_model = max(metrics_data, key=lambda x: float(x['R² Score']))
            st.metric("Best R² Score", best_model['R² Score'])
    
    # Model performance
    st.subheader("🤖 Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Feature importance
    if feature_importance_data:
        st.subheader("🔍 Feature Importance (Random Forest)")
        importance_df = pd.DataFrame(feature_importance_data)
        importance_df = importance_df.sort_values('Importance', ascending=False)
        st.dataframe(importance_df, use_container_width=True)
        
        # Feature importance chart
        chart_data = importance_df.head(10).copy()
        chart_data['Importance'] = chart_data['Importance'].astype(float)
        st.bar_chart(chart_data.set_index('Feature')['Importance'])
    
    # Actual vs Predicted prices
    st.subheader("📈 Actual vs Predicted Prices")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Date': symbol_data.iloc[y_test.index]['date']
    })
    
    for model_name, y_pred in predictions.items():
        comparison_df[model_name] = y_pred
    
    comparison_df = comparison_df.set_index('Date').sort_index()
    
    # Display chart
    st.line_chart(comparison_df)
    
    # Prediction details
    st.subheader("🔮 Recent Predictions")
    recent_predictions = comparison_df.tail(10).reset_index()
    st.dataframe(recent_predictions.style.format({
        'Actual': '${:.2f}',
        'Linear Regression': '${:.2f}',
        'Random Forest': '${:.2f}'
    }), use_container_width=True)
    
    # Technical Analysis
    st.subheader("📊 Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Moving averages
        st.write("**Moving Averages**")
        ma_data = symbol_data[['date', 'close_price', 'sma_5', 'sma_10', 'sma_20']].set_index('date').tail(50)
        st.line_chart(ma_data)
    
    with col2:
        # Volume analysis
        st.write("**Volume Analysis**")
        volume_data = symbol_data[['date', 'volume', 'volume_ma_5']].set_index('date').tail(50)
        st.area_chart(volume_data)
    
    # Correlation Analysis
    st.subheader("🔗 Feature Correlation with Next Day Price")
    
    # Calculate correlations
    correlation_data = []
    for feature in selected_features:
        corr = symbol_data[feature].corr(symbol_data['next_day_price'])
        correlation_data.append({
            'Feature': feature,
            'Correlation': f"{corr:.4f}",
            'Strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
        })
    
    correlation_df = pd.DataFrame(correlation_data).sort_values('Correlation', key=lambda x: x.abs(), ascending=False)
    st.dataframe(correlation_df, use_container_width=True)
    
    # Prediction for next day
    st.subheader("🎯 Next Day Prediction")
    
    if st.button("Generate Next Day Forecast"):
        # Use the most recent data point for prediction
        latest_data = symbol_data.iloc[-1:][selected_features]
        latest_data = latest_data.fillna(latest_data.mean())
        
        next_day_predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(latest_data)[0]
            next_day_predictions[model_name] = prediction
        
        col1, col2, col3 = st.columns(3)
        with col1:
            current = symbol_data['close_price'].iloc[-1]
            st.metric("Current Price", f"${current:.2f}")
        
        for i, (model_name, prediction) in enumerate(next_day_predictions.items()):
            change = prediction - current
            change_pct = (change / current) * 100
            
            with [col2, col3][i]:
                st.metric(
                    f"{model_name} Prediction",
                    f"${prediction:.2f}",
                    f"{change:+.2f} ({change_pct:+.1f}%)"
                )
    
else:
    st.warning("Please select at least one feature group for prediction.")

# Data overview
st.sidebar.subheader("📋 Data Overview")
st.sidebar.write(f"**Total Records:** {len(df):,}")
st.sidebar.write(f"**Stocks Available:** {len(symbols)}")
st.sidebar.write(f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Footer
st.markdown("---")
st.markdown("**MADSC102 Final Project** | Multi-Feature Stock Price Prediction | Zubair Ilyas")
