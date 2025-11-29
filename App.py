# Simple version without plotly
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Price Dashboard")

@st.cache_data
def load_data():
    try:
        with open('stock_data.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return generate_fallback_data()

def generate_fallback_data():
    # Simple data generation
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    symbols = ['MSFT', 'AAPL', 'GOOGL']
    
    data = []
    for symbol in symbols:
        prices = 150 + np.cumsum(np.random.normal(0, 2, len(dates)))
        for i, date in enumerate(dates):
            data.append({
                'date': date, 'symbol': symbol,
                'close_price': max(prices[i], 10),
                'volume': np.random.randint(1000000, 50000000)
            })
    return pd.DataFrame(data)

df = load_data()
symbol = st.selectbox("Select Stock", df['symbol'].unique())
filtered_df = df[df['symbol'] == symbol]

st.metric("Current Price", f"${filtered_df['close_price'].iloc[-1]:.2f}")
st.line_chart(filtered_df.set_index('date')['close_price'])
st.dataframe(filtered_df.tail(10))