# Enhanced Stock Prediction with Multiple Features (No scikit-learn required)
import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
            if i >= 20:  # Need enough history for indicators
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

# Custom Linear Regression implementation
class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: (X'X)^-1 X'y
        try:
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            self.intercept = coefficients[0]
            self.coefficients = coefficients[1:]
        except:
            # Fallback if matrix is singular
            self.intercept = np.mean(y)
            self.coefficients = np.zeros(X.shape[1])
    
    def predict(self, X):
        if self.coefficients is None:
            return np.full(X.shape[0], np.mean(y))
        return self.intercept + X @ self.coefficients

# Custom Random Forest-like simple ensemble
class SimpleEnsemble:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
    
    def fit(self, X, y):
        self.models = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_models):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            # Train simple model (average of feature contributions)
            model = SimpleLinearRegression()
            model.fit(X_sample, y_sample)
            self.models.append(model)
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model in self.models:
            predictions += model.predict(X)
        return predictions / len(self.models)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Accuracy': max(0, (1 - mae / np.mean(y_true)) * 100)
    }

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
    ["Linear Regression", "Simple Ensemble", "Both"]
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
    
    # Simple train-test split (last 20% for testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to numpy for custom models
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    
    # Train models
    models = {}
    predictions = {}
    
    if model_type in ["Linear Regression", "Both"]:
        lr_model = SimpleLinearRegression()
        lr_model.fit(X_train_np, y_train_np)
        models['Linear Regression'] = lr_model
        predictions['Linear Regression'] = lr_model.predict(X_test_np)
    
    if model_type in ["Simple Ensemble", "Both"]:
        ensemble_model = SimpleEnsemble(n_models=10)
        ensemble_model.fit(X_train, y_train)
        models['Simple Ensemble'] = ensemble_model
        predictions['Simple Ensemble'] = ensemble_model.predict(X_test_np)
    
    # Calculate metrics
    metrics_data = []
    
    for model_name, y_pred in predictions.items():
        metrics = calculate_metrics(y_test_np, y_pred)
        
        metrics_data.append({
            'Model': model_name,
            'MAE': f"${metrics['MAE']:.2f}",
            'RMSE': f"${metrics['RMSE']:.2f}",
            'R² Score': f"{metrics['R²']:.4f}",
            'Accuracy': f"{metrics['Accuracy']:.1f}%"
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
    
    # Actual vs Predicted prices
    st.subheader("📈 Actual vs Predicted Prices")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual': y_test_np,
        'Date': symbol_data.iloc[y_test.index]['date'].values
    })
    
    for model_name, y_pred in predictions.items():
        comparison_df[model_name] = y_pred
    
    comparison_df = comparison_df.set_index('Date').sort_index()
    
    # Display chart
    st.line_chart(comparison_df)
    
    # Prediction details
    st.subheader("🔮 Recent Predictions")
    recent_predictions = comparison_df.tail(10).reset_index()
    
    # Format the display
    display_df = recent_predictions.copy()
    for col in ['Actual', 'Linear Regression', 'Simple Ensemble']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)
    
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
    
    # Feature Correlation Analysis
    st.subheader("🔗 Feature Correlation with Next Day Price")
    
    # Calculate correlations
    correlation_data = []
    for feature in selected_features:
        if feature in symbol_data.columns:
            corr = symbol_data[feature].corr(symbol_data['next_day_price'])
            correlation_data.append({
                'Feature': feature,
                'Correlation': f"{corr:.4f}",
                'Strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            })
    
    if correlation_data:
        correlation_df = pd.DataFrame(correlation_data).sort_values('Correlation', key=lambda x: pd.to_numeric(x.str.replace(r'[^\d.-]', '', regex=True)), ascending=False)
        st.dataframe(correlation_df, use_container_width=True)
    
    # Next Day Prediction
    st.subheader("🎯 Next Day Prediction")
    
    if st.button("Generate Next Day Forecast"):
        # Use the most recent data point for prediction
        if len(symbol_data) > 0:
            latest_data = symbol_data[selected_features].iloc[-1:].values
            
            next_day_predictions = {}
            for model_name, model in models.items():
                try:
                    prediction = model.predict(latest_data)[0]
                    next_day_predictions[model_name] = prediction
                except:
                    next_day_predictions[model_name] = symbol_data['close_price'].iloc[-1]
            
            current = symbol_data['close_price'].iloc[-1]
            
            cols = st.columns(len(next_day_predictions) + 1)
            
            with cols[0]:
                st.metric("Current Price", f"${current:.2f}")
            
            for i, (model_name, prediction) in enumerate(next_day_predictions.items()):
                change = prediction - current
                change_pct = (change / current) * 100
                
                with cols[i + 1]:
                    st.metric(
                        f"{model_name}",
                        f"${prediction:.2f}",
                        f"{change:+.2f} ({change_pct:+.1f}%)"
                    )
    
    # Feature Importance (simplified)
    st.subheader("📊 Feature Impact Analysis")
    
    if 'Linear Regression' in models:
        lr_model = models['Linear Regression']
        if lr_model.coefficients is not None:
            importance_data = []
            for feature, coef in zip(selected_features, lr_model.coefficients):
                importance_data.append({
                    'Feature': feature,
                    'Coefficient': f"{coef:.4f}",
                    'Impact': 'Positive' if coef > 0 else 'Negative'
                })
            
            importance_df = pd.DataFrame(importance_data).sort_values('Coefficient', key=lambda x: pd.to_numeric(x.str.replace(r'[^\d.-]', '', regex=True)), ascending=False)
            st.dataframe(importance_df, use_container_width=True)

else:
    st.warning("Please select at least one feature group for prediction.")

# Data overview
st.sidebar.subheader("📋 Data Overview")
st.sidebar.write(f"**Total Records:** {len(df):,}")
st.sidebar.write(f"**Stocks Available:** {len(symbols)}")
st.sidebar.write(f"**Features Available:** {len(selected_features)}")

# Footer
st.markdown("---")
st.markdown("**MADSC102 Final Project** | Multi-Feature Stock Price Prediction | Zubair Ilyas")

# Add some explanation
with st.expander("ℹ️ About This Prediction System"):
    st.markdown("""
    **How It Works:**
    - **Linear Regression**: Uses statistical relationships between features and stock prices
    - **Simple Ensemble**: Combines multiple models for more robust predictions
    - **Technical Indicators**: Moving averages, momentum, volatility measures
    - **Market Factors**: Sentiment, economic indicators
    
    **Key Features Used:**
    - Price history and patterns
    - Volume trends
    - Technical indicators
    - Market sentiment
    - Economic factors
    
    **Note**: This uses custom ML implementations (no scikit-learn required)
    """)
