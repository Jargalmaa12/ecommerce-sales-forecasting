import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Forecasting models
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("your_data.csv")   # Replace with your dataset
    df['ds'] = pd.to_datetime(df['date'])  # Prophet expects 'ds'
    df['y'] = df['sales']                # Prophet expects 'y'
    return df

df = load_data()

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["Validation (Train/Test Split)", "Forecast Future"])
model_choice = st.sidebar.selectbox("Select Model:", ["Prophet", "ARIMA", "XGBoost"])
horizon = st.sidebar.slider("Forecast horizon (days):", 30, 180, 90)

st.title("🔮 Future Forecast Mode (Beyond Dataset)")

# --- Split Data ---
train = df.iloc[:-horizon]
test = df.iloc[-horizon:]

# --- Forecast Functions ---
def forecast_prophet(train, horizon):
    m = Prophet()
    m.fit(train[['ds', 'y']])
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

def forecast_arima(train, horizon):
    model = ARIMA(train['y'], order=(5,1,0))  # You can tune order
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    future_dates = pd.date_range(start=train['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({'ds': future_dates, 'yhat': forecast})

def forecast_xgboost(train, horizon):
    # Prepare features: here we just use lag features for demo
    df_xgb = train.copy()
    df_xgb['lag1'] = df_xgb['y'].shift(1)
    df_xgb = df_xgb.dropna()

    X = df_xgb[['lag1']]
    y = df_xgb['y']

    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)

    preds = []
    last_val = df_xgb['y'].iloc[-1]

    for _ in range(horizon):
        pred = model.predict(np.array([[last_val]]))
        preds.append(pred[0])
        last_val = pred[0]

    future_dates = pd.date_range(start=train['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({'ds': future_dates, 'yhat': preds})

# --- Run Forecast ---
if model_choice == "Prophet":
    forecast = forecast_prophet(train, horizon)
elif model_choice == "ARIMA":
    forecast = forecast_arima(train, horizon)
else:
    forecast = forecast_xgboost(train, horizon)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['ds'], df['y'], label="Actual", color="blue")
ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color="orange")

if 'yhat_lower' in forecast.columns:  # Prophet only
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="orange", alpha=0.2)

ax.legend()
st.pyplot(fig)

# --- Insights ---
if mode == "Forecast Future":
    st.info(f"📈 Forecasting {horizon} days beyond {df['ds'].iloc[-1].date()} using {model_choice}")
else:
    if model_choice == "Prophet":
        y_true = test['y'].values
        y_pred = forecast['yhat'].iloc[-horizon:].values
    else:
        y_true = test['y'].values
        y_pred = forecast['yhat'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    st.success(f"✅ {model_choice} Validation RMSE: {rmse:.2f}")
