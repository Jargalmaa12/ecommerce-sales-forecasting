# 📊 FMCG Sales Forecasting Dashboard (Prophet, ARIMA, XGBoost + Comparison)

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Models
import statsmodels.api as sm
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# --- Load dataset ---
data_path = os.path.join("data", "FMCG_2022_2024.csv")
df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'])
df['revenue'] = df['units_sold'] * df['price_unit']

daily = (
    df.groupby('date')['revenue']
    .sum()
    .reset_index()
    .rename(columns={'revenue':'daily_revenue'})
)

# --- Streamlit UI ---
st.title("🛒 FMCG Sales Forecasting Dashboard")
st.write("Forecast future sales using **Prophet, ARIMA, or XGBoost** models.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA", "XGBoost", "Compare All"])
horizon = st.sidebar.slider("Forecast horizon (days):", 30, 180, 90)

# Train/test split
train = daily.iloc[:-horizon]
test = daily.iloc[-horizon:]

# Evaluation function
def evaluate(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Run Selected Model ---
results = {}

if model_choice == "Prophet":
    prophet_df = train.rename(columns={'date':'ds','daily_revenue':'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    forecast_prophet = forecast[['ds','yhat']].set_index('ds').loc[test['date']]
    results['Prophet'] = evaluate(test['daily_revenue'], forecast_prophet['yhat'])

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train['date'], train['daily_revenue'], label="Train")
    ax.plot(test['date'], test['daily_revenue'], label="Test", color="black")
    ax.plot(forecast_prophet.index, forecast_prophet['yhat'], label="Prophet Forecast", color="red")
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, color="red")
    ax.legend()
    st.pyplot(fig)

elif model_choice == "ARIMA":
    model = sm.tsa.ARIMA(train['daily_revenue'], order=(5,1,2))
    arima_fit = model.fit()
    forecast_arima = arima_fit.forecast(steps=horizon)
    results['ARIMA'] = evaluate(test['daily_revenue'], forecast_arima)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train['date'], train['daily_revenue'], label="Train")
    ax.plot(test['date'], test['daily_revenue'], label="Test", color="black")
    ax.plot(test['date'], forecast_arima, label="ARIMA Forecast", color="red")
    ax.legend()
    st.pyplot(fig)

elif model_choice == "XGBoost":
    lag_df = daily.copy()
    for lag in [1,7,30]:
        lag_df[f"lag_{lag}"] = lag_df['daily_revenue'].shift(lag)

    lag_df = lag_df.dropna()
    train_lag = lag_df.iloc[:-horizon]
    test_lag = lag_df.iloc[-horizon:]

    X_train, y_train = train_lag.drop(['date','daily_revenue'], axis=1), train_lag['daily_revenue']
    X_test, y_test = test_lag.drop(['date','daily_revenue'], axis=1), test_lag['daily_revenue']

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
    xgb.fit(X_train, y_train)
    forecast_xgb = xgb.predict(X_test)

    results['XGBoost'] = evaluate(y_test, forecast_xgb)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train['date'], train['daily_revenue'], label="Train")
    ax.plot(test['date'], test['daily_revenue'], label="Test", color="black")
    ax.plot(test_lag['date'], forecast_xgb, label="XGBoost Forecast", color="red")
    ax.legend()
    st.pyplot(fig)

# --- Compare All Models ---
elif model_choice == "Compare All":
    # Prophet
    prophet_df = train.rename(columns={'date':'ds','daily_revenue':'y'})
    model_p = Prophet(yearly_seasonality=True)
    model_p.fit(prophet_df)
    future = model_p.make_future_dataframe(periods=horizon)
    forecast_p = model_p.predict(future)
    forecast_prophet = forecast_p[['ds','yhat']].set_index('ds').loc[test['date']]
    results['Prophet'] = evaluate(test['daily_revenue'], forecast_prophet['yhat'])

    # ARIMA
    model_a = sm.tsa.ARIMA(train['daily_revenue'], order=(5,1,2))
    arima_fit = model_a.fit()
    forecast_arima = arima_fit.forecast(steps=horizon)
    results['ARIMA'] = evaluate(test['daily_revenue'], forecast_arima)

    # XGBoost
    lag_df = daily.copy()
    for lag in [1,7,30]:
        lag_df[f"lag_{lag}"] = lag_df['daily_revenue'].shift(lag)
    lag_df = lag_df.dropna()
    train_lag = lag_df.iloc[:-horizon]
    test_lag = lag_df.iloc[-horizon:]
    X_train, y_train = train_lag.drop(['date','daily_revenue'], axis=1), train_lag['daily_revenue']
    X_test, y_test = test_lag.drop(['date','daily_revenue'], axis=1), test_lag['daily_revenue']
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
    xgb.fit(X_train, y_train)
    forecast_xgb = xgb.predict(X_test)
    results['XGBoost'] = evaluate(y_test, forecast_xgb)

    # Show comparison table
    st.subheader("📊 Model Comparison")
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df)

    # Optional: Bar plot
    st.bar_chart(results_df['RMSE'])

# --- Show metrics for single model ---
if model_choice != "Compare All":
    st.subheader("📈 Model Performance")
    for k, v in results[list(results.keys())[0]].items():
        st.write(f"**{k}:** {v:.2f}")
