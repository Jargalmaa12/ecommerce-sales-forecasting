import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

# --- Load dataset ---
data_path = "data/FMCG_2022_2024.csv"  # adjust if needed
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date")

# Aggregate daily
daily = df.groupby("date")["units_sold"].sum().reset_index()
daily = daily.rename(columns={"date":"ds", "units_sold":"y"})

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["Validation (Train/Test Split)", "Forecast Future"])
model_choice = st.sidebar.selectbox("Select Model:", ["Prophet", "SARIMA", "XGBoost"])
horizon = st.sidebar.slider("Forecast horizon (days):", 30, 180, 90)

# --- Helper function ---
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# ==============================
# --- Validation Mode ---
# ==============================
if mode == "Validation (Train/Test Split)":
    st.subheader("📊 Validation Mode (Train/Test Split)")
    split_date = daily["ds"].max() - pd.Timedelta(days=horizon)
    train = daily[daily["ds"] <= split_date]
    test = daily[daily["ds"] > split_date]

    # --- Prophet ---
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

    # --- SARIMA ---
    elif model_choice == "SARIMA":
        sarima_model = SARIMAX(train["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_fit = sarima_model.fit(disp=False)
        preds = sarima_fit.forecast(steps=len(test))
        rmse, mae, mape = evaluate(test["y"].values, preds)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test")
        ax.plot(test["ds"], preds, label="SARIMA Forecast", color="red")
        ax.legend()
        st.pyplot(fig)

    # --- XGBoost ---
    elif model_choice == "XGBoost":
        lagged = daily.copy()
        for lag in [1,7,30]:
            lagged[f"lag_{lag}"] = lagged["y"].shift(lag)
        lagged["rolling_7"] = lagged["y"].rolling(7).mean()
        lagged["rolling_30"] = lagged["y"].rolling(30).mean()
        lagged["dayofweek"] = lagged["ds"].dt.dayofweek
        lagged["month"] = lagged["ds"].dt.month
        lagged["quarter"] = lagged["ds"].dt.quarter
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
        ax.plot(test_l["ds"], preds, label="XGBoost Forecast", color="green")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📉 Model Performance")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

# ==============================
# --- Future Forecast Mode ---
# ==============================
else:
    st.subheader("🔮 Future Forecast Mode (Beyond Dataset)")

    # --- Prophet ---
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

    # --- SARIMA ---
    elif model_choice == "SARIMA":
        sarima_model = SARIMAX(daily["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
        sarima_fit = sarima_model.fit(disp=False)
        preds = sarima_fit.forecast(steps=horizon)
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, preds, label="SARIMA Forecast", color="red")
        ax.legend()
        st.pyplot(fig)

    # --- XGBoost ---
    elif model_choice == "XGBoost":
        lagged = daily.copy()
        for lag in [1,7,30]:
            lagged[f"lag_{lag}"] = lagged["y"].shift(lag)
        lagged["rolling_7"] = lagged["y"].rolling(7).mean()
        lagged["rolling_30"] = lagged["y"].rolling(30).mean()
        lagged["dayofweek"] = lagged["ds"].dt.dayofweek
        lagged["month"] = lagged["ds"].dt.month
        lagged["quarter"] = lagged["ds"].dt.quarter
        lagged = lagged.dropna()

        X = lagged.drop(["ds","y"], axis=1)
        y = lagged["y"]
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
        model.fit(X, y)

        future_preds = []
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)
        last_known = lagged.iloc[-1:].copy()

        for date in future_dates:
            row = {
                "lag_1": last_known["y"].values[-1],
                "lag_7": last_known["lag_1"].values[-1],
                "lag_30": last_known["lag_7"].values[-1],
                "rolling_7": last_known["rolling_7"].values[-1],
                "rolling_30": last_known["rolling_30"].values[-1],
                "dayofweek": date.dayofweek,
                "month": date.month,
                "quarter": date.quarter
            }
            row_df = pd.DataFrame([row])
            pred = model.predict(row_df)[0]
            future_preds.append(pred)

            # update recursion
            last_known["y"] = [pred]
            last_known["lag_1"] = pred
            last_known["rolling_7"] = ((last_known["rolling_7"].values[-1] * 6) + pred) / 7
            last_known["rolling_30"] = ((last_known["rolling_30"].values[-1] * 29) + pred) / 30

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, future_preds, label="XGBoost Forecast", color="green")
        ax.legend()
        st.pyplot(fig)

    # Info
    st.info(f"Forecasting **{horizon} days beyond {daily['ds'].max().date()}** using {model_choice}")
