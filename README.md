# 🛒 E-Commerce Sales Forecasting  

## 📌 Overview  
This project applies time-series and machine learning models to forecast daily sales for an e-commerce / FMCG platform.  
The goal is to help businesses make smarter decisions around **inventory planning, marketing campaigns, and revenue management** by predicting future demand patterns.  

## 🎯 Goals  
- Transform raw sales data into a clean, structured dataset  
- Detect **seasonality, spikes, and anomalies** using exploratory analysis  
- Train forecasting models: **ARIMA, Prophet, and XGBoost**  
- Compare performance with metrics (**RMSE, MAE, MAPE**)  
- Build an **interactive Streamlit dashboard** for testing models and forecast horizons  
- Convert model results into **practical business insights**  

## 📂 Project Layout  
ecommerce-sales-forecasting/
│── data/ # raw and processed datasets
│── notebooks/ # EDA, feature engineering, and model training
│── dashboard/ # Streamlit application
│── visuals/ # plots and screenshots
│── requirements.txt
│── README.md

markdown
Copy code

## 🛠 Tools  
- **Python**: pandas, numpy, scikit-learn, statsmodels, XGBoost, Prophet  
- **Visualization**: matplotlib, seaborn, plotly  
- **Deployment**: Streamlit  
- **Version Control**: GitHub  

## 📊 Workflow  
1. **Data Preparation** – Aggregate and clean sales data  
2. **Exploratory Data Analysis** – Identify seasonality and demand trends  
3. **Modeling** – Build ARIMA, Prophet, and XGBoost forecasts  
4. **Dashboard** – Deploy interactive forecasts with adjustable horizons  
5. **Insights** – Translate results into business recommendations  

## 🔮 Model Insights  
- **Prophet**: Captures long-term trends and seasonality well.  
- **ARIMA**: Good for short-term forecasting, weaker at seasonal patterns.  
- **XGBoost**: Strong validation accuracy but future forecasts tend to flatten without engineered features.  

**Takeaway:**  
- Use **Prophet** for long-term planning.  
- Use **XGBoost** for short-term demand spikes.  
- Combining both could yield the most reliable results.  

## 📈 Dashboard  
The Streamlit app allows users to:  
- Select forecasting models (Prophet / ARIMA / XGBoost)  
- Adjust forecast horizon (30–180 days)  
- Compare error metrics (RMSE, MAE, MAPE)  
- Switch between **validation mode** and **future mode**  

![Dashboard Screenshot](visuals/Screenshot_2025-10-02.png)  

👉 **Live App:** [E-Commerce Forecasting Dashboard](https://ecommerce-sales-forecasting-muvi4tfwfefncf77qxlnvt.streamlit.app/)  

## 🚀 Status  
- ✅ Models trained and dashboard deployed  
- 🔄 Next: test hybrid models and add holiday/promotion effects  

## 📌 Key Learnings  
- Built a complete **ML forecasting pipeline**  
- Compared traditional time-series and ML models  
- Learned trade-offs between **accuracy vs interpretability**  
- Deployed results in an **interactive app** for business use  
