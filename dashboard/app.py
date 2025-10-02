import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import statsmodels.api as sm
from xgboost import XGBRegressor
import datetime

# --- Load data ---
data_path = "data/FMCG_2022_2024.csv"  # adjust if different
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date")
daily = df.groupby("date")["units_sold"].sum().reset_index()
daily = daily.rename(columns={"date":"ds", "units_sold":"y"})  # Prophet requires ds, y

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["Validation (Train/Test Split)", "Forecast Future"])
model_choice = st.sidebar.selectbox("Select Model:", ["Prophet", "ARIMA", "XGBoost"])
horizon = st.sidebar.slider("Forecast horizon (days):", 30, 180, 90)

# --- Helper functions ---
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# --- Validation Mode ---
if mode == "Validation (Train/Test Split)":
    st.subheader("📊 Validation Mode (Train/Test Split)")
    split_date = daily["ds"].max() - pd.Timedelta(days=horizon)
    train = daily[daily["ds"] <= split_date]
    test = daily[daily["ds"] > split_date]

    if model_choice == "Prophet":
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        preds = forecast.set_index("ds").loc[test["ds"], "yhat"]
        rmse, mae, mape = evaluate(test["y"], preds)

        st.line_chart(pd.DataFrame({"Train":train.set_index("ds")["y"],
                                    "Test":test.set_index("ds")["y"],
                                    "Forecast":preds}))

    elif model_choice == "ARIMA":
        arima_model = sm.tsa.ARIMA(train["y"], order=(5,1,2))
        arima_fit = arima_model.fit()
        preds = arima_fit.forecast(steps=len(test))
        rmse, mae, mape = evaluate(test["y"].values, preds)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test")
        ax.plot(test["ds"], preds, label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "XGBoost":
        # Feature engineering (lag features)
        lagged = daily.copy()
        lagged["lag1"] = lagged["y"].shift(1)
        lagged = lagged.dropna()

        train = lagged[lagged["ds"] <= split_date]
        test = lagged[lagged["ds"] > split_date]

        model = XGBRegressor(n_estimators=100)
        model.fit(train[["lag1"]], train["y"])
        preds = model.predict(test[["lag1"]])
        rmse, mae, mape = evaluate(test["y"].values, preds)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test")
        ax.plot(test["ds"], preds, label="XGBoost Forecast")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📉 Model Performance")
    col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")


# --- Future Forecast Mode ---
else:
    st.subheader("🔮 Future Forecast Mode (Beyond Dataset)")
    if model_choice == "Prophet":
        model = Prophet()
        model.fit(daily)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(forecast["ds"], forecast["yhat"], label="Forecast")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "ARIMA":
        arima_model = sm.tsa.ARIMA(daily["y"], order=(5,1,2))
        arima_fit = arima_model.fit()
        preds = arima_fit.forecast(steps=horizon)

        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, preds, label="ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "XGBoost":
        lagged = daily.copy()
        lagged["lag1"] = lagged["y"].shift(1)
        lagged = lagged.dropna()

        train = lagged.copy()
        model = XGBRegressor(n_estimators=100)
        model.fit(train[["lag1"]], train["y"])

        preds = []
        last_value = train["y"].iloc[-1]
        for _ in range(horizon):
            next_pred = model.predict([[last_value]])[0]
            preds.append(next_pred)
            last_value = next_pred

        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, preds, label="XGBoost Forecast")
        ax.legend()
        st.pyplot(fig)

    st.info(f"Forecasting **{horizon} days beyond {daily['ds'].max().date()}** using {model_choice}")
