# 📊 Simple FMCG Sales Forecasting Dashboard (Prophet only)

import streamlit as st
import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Load dataset ---
data_path = "FMCG_2022_2024.csv"  # put this file in the same folder as app.py
df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'])
df['revenue'] = df['units_sold'] * df['price_unit']

daily = df.groupby('date')['revenue'].sum().reset_index().rename(columns={'revenue':'daily_revenue'})

# --- Streamlit UI ---
st.title("🛒 FMCG Sales Forecasting (Prophet Demo)")
st.write("Forecast future sales using **Prophet** model.")

# Forecast horizon
horizon = st.slider("Select forecast horizon (days):", 30, 180, 90)

# Prepare Prophet
prophet_df = daily.rename(columns={'date':'ds','daily_revenue':'y'})
model = Prophet(yearly_seasonality=True)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# --- Plot ---
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(daily['date'], daily['daily_revenue'], label="Actual")
ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color="red")
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                alpha=0.3, label="Confidence Interval")
ax.legend()
st.pyplot(fig)
