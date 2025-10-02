# --- Future Forecast Mode ---
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

    # --- XGBoost with Lag + Calendar Features ---
    elif model_choice == "XGBoost":
        lagged = daily.copy()
        for lag in [1,7,30]:
            lagged[f"lag_{lag}"] = lagged["y"].shift(lag)

        lagged["rolling_7"] = lagged["y"].rolling(7).mean()
        lagged["rolling_30"] = lagged["y"].rolling(30).mean()

        # Add calendar features
        lagged["dayofweek"] = lagged["ds"].dt.dayofweek
        lagged["month"] = lagged["ds"].dt.month
        lagged["quarter"] = lagged["ds"].dt.quarter

        lagged = lagged.dropna()

        # Train model
        X = lagged.drop(["ds","y"], axis=1)
        y = lagged["y"]
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
        model.fit(X, y)

        # Recursive future forecast
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

            # update rolling windows
            last_known["y"] = [pred]
            last_known["lag_1"] = pred
            last_known["rolling_7"] = ((last_known["rolling_7"].values[-1] * 6) + pred) / 7
            last_known["rolling_30"] = ((last_known["rolling_30"].values[-1] * 29) + pred) / 30

        # Plot
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(daily["ds"], daily["y"], label="Actual")
        ax.plot(future_dates, future_preds, label="XGBoost Forecast", color="green")
        ax.legend()
        st.pyplot(fig)

    # Info message
    st.info(f"Forecasting **{horizon} days beyond {daily['ds'].max().date()}** using {model_choice}")
