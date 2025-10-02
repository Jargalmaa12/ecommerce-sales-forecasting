import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

# --- Load dataset ---
data_path = "data/FMCG_2022_2024.csv"  # adjust if different
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date")
daily = df.groupby("date")["units_sold"].sum().reset_index()
daily = daily.rename(columns={"date":"ds", "units_sold":"y"})

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["Validation (Train/Test Split)", "Forecast Future"])
model_choice = st.sidebar.selectbox("Select Model:", ["Prophet", "SARIMA", "XGBoost"])
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

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test")
        ax.plot(test["ds"], preds, label="Prophet Forecast")
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "SARIMA":
        sarima_model = SARIMAX(train["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_fit = sarima_model.fit(disp=False)
        preds = sarima_fit.forecast(steps=len(test))
        rmse, mae, mape = evaluate(test["y"].values, preds)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test")
        ax.plot(test["ds"], preds, label="SARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "XGBoost":
        lagged = daily.copy()
        for lag in [1,7,30]:
            lagged[f"lag_{lag}"] = lagged["y"].shift(lag)
        lagged["rolling_7"] = lagged["y"].rolling(7).mean()
        lagged["rolling_30"] = lagged["y"].rolling(30).mean()
        lagged = lagged.dropna()

        train_l = lagged[lagged["ds"] <= split_date]
        test_l = lagged[lagged["ds"] > split_date]

        X_train, y_train = train_l.drop(["ds","y"], axis=1), train_l["y"]
        X_test, y_test = test_l.drop(["ds","y"], axis=1), test_l["y"]

        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse, mae, mape = evaluate(y_test.values, preds)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train_l["ds"], y_train, label="Train")
        ax.plot(test_l["ds"], y_test, label="Test")
        ax.plot(test_l["ds"], preds, label="XGBoost Forecast")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📉 Model Performance")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

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
        ax.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "SARIMA":
        sarima_model = SARIMAX(daily["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_fit = sarima_model.fit(disp=False)
        preds = sarima_fit.forecast(steps=horizon)
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, preds, label="SARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "XGBoost":
        lagged = daily.copy()
        for lag in [1,7,30]:
            lagged[f"lag_{lag}"] = lagged["y"].shift(lag)
        lagged["rolling_7"] = lagged["y"].rolling(7).mean()
        lagged["rolling_30"] = lagged["y"].rolling(30).mean()
        lagged = lagged.dropna()

        X = lagged.drop(["ds","y"], axis=1)
        y = lagged["y"]
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
        model.fit(X, y)

        # Recursive future forecast
        last_row = lagged.iloc[-1].copy()
        future_preds = []
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)

        for _ in range(horizon):
            row = pd.DataFrame([{
                "lag_1": last_row["y"],
                "lag_7": last_row["lag_1"],  # approximate recursion
                "lag_30": last_row["lag_7"], # approximate recursion
                "rolling_7": last_row["rolling_7"],
                "rolling_30": last_row["rolling_30"]
            }])
            pred = model.predict(row)[0]
            future_preds.append(pred)
            last_row["y"] = pred
            last_row["lag_1"] = pred

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, future_preds, label="XGBoost Forecast")
        ax.legend()
        st.pyplot(fig)

    st.info(f"Forecasting **{horizon} days beyond {daily['ds'].max().date()}** using {model_choice}")
