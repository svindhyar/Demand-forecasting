# Demand-forecasting
Time series forecasting for retail demand using XGBoost and LightGBM. Predicts item-level sales across 10 stores and 50 items using historical data (2013-2017) with advanced feature engineering and lag variables.

# 📊 Store Item Demand Forecasting

A comprehensive machine learning solution for predicting retail item demand across multiple store locations using time series analysis and gradient boosting algorithms.

## 🎯 Project Overview

This project tackles the challenge of forecasting 3 months of item-level sales data for 10 different stores, each carrying 50 items. Using 5 years of historical sales data (2013-2017), we employ advanced feature engineering and state-of-the-art machine learning models to deliver accurate demand predictions.

## 📁 Dataset

- **Training Data**: Historical sales (2013-2017)
- **Test Data**: 3 months requiring predictions
- **Stores**: 10 locations
- **Items**: 50 unique items per store
- **Total Combinations**: 500 store-item pairs

## 🔧 Key Features

- **Temporal Features**: Year, month, day, day of week, quarter, week of year
- **Cyclical Encoding**: Sine/cosine transformations for periodic patterns
- **Lag Features**: 7, 14, 28, 30, 60, and 90-day lags
- **Rolling Statistics**: Mean and standard deviation over 7, 14, 30, and 60-day windows
- **Exponential Weighted Averages**: 7 and 30-day spans

## 🤖 Models Implemented

- **XGBoost**: Gradient boosting with histogram-based tree learning
- **LightGBM**: High-performance gradient boosting framework

## 📊 Results

Models evaluated using RMSE, MAE, R², and MAPE metrics with time-based train-validation split (80-20).

## 🛠️ Technologies

- Python 3.10+
- pandas, numpy
- scikit-learn
- XGBoost, LightGBM
- matplotlib, seaborn, plotly
- statsmodels

## 📈 Visualizations

- Time series decomposition (trend, seasonality, residuals)
- ACF/PACF analysis
- Store and item performance comparison
- Seasonal patterns and day-of-week trends
- Model performance comparison
- Residual analysis
