elif model_choice == "XGBoost":
    # --- Feature Engineering ---
    lagged = daily.copy()
    for lag in [1, 7, 30]:
        lagged[f"lag_{lag}"] = lagged["y"].shift(lag)

    lagged["rolling_7"] = lagged["y"].rolling(7).mean()
    lagged["rolling_30"] = lagged["y"].rolling(30).mean()

    # Add calendar features for seasonality
    lagged["dayofweek"] = lagged["ds"].dt.dayofweek
    lagged["month"] = lagged["ds"].dt.month
    lagged["quarter"] = lagged["ds"].dt.quarter

    lagged = lagged.dropna()

    # Train model
    X = lagged.drop(["ds", "y"], axis=1)
    y = lagged["y"]
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model.fit(X, y)

    # --- Recursive Future Forecast ---
    future_preds = []
    future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=horizon)

    # start with the last known row
    last_known = lagged.iloc[-1:].copy()

    for date in future_dates:
        row = {
            "lag_1": last_known["y"].values[-1],
            "lag_7": last_known["lag_1"].values[-1],   # shift yesterday → last week approx
            "lag_30": last_known["lag_7"].values[-1],  # shift last week → last month approx
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

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(daily["ds"], daily["y"], label="Actual")
    ax.plot(future_dates, future_preds, label="XGBoost Forecast", color="green")
    ax.legend()
    st.pyplot(fig)

    st.success(f"✅ Forecasting {horizon} days beyond {daily['ds'].max().date()} using XGBoost with lag + calendar features")
