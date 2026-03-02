"""
Store Item Demand Forecasting - Training Script
Automated training pipeline for demand forecasting models
"""

import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime

# Machine Learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    print(f"✓ Training data loaded: {df_train.shape}")
    print(f"✓ Test data loaded: {df_test.shape}")
    
    return df_train, df_test

def create_features(df):
    """Create comprehensive features for demand forecasting"""
    df = df.copy()
    
    # Temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def create_lag_features(df, lags=[7, 14, 28, 30, 60, 90]):
    """Create lag features and rolling statistics"""
    df = df.copy()
    df = df.sort_values(['store', 'item', 'date'])
    
    if 'sales' in df.columns:
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30, 60]:
            df[f'rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
        
        # Exponential weighted mean
        df['ewm_7'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).ewm(span=7, adjust=False).mean()
        )
        df['ewm_30'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).ewm(span=30, adjust=False).mean()
        )
    
    return df

def prepare_features(df_train, df_test):
    """Prepare features for modeling"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Create temporal features
    df_train = create_features(df_train)
    df_test = create_features(df_test)
    print(f"✓ Temporal features created")
    
    # Create lag features
    df_train_with_lags = create_lag_features(df_train)
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    df_combined = create_lag_features(df_combined)
    df_test_with_lags = df_combined[df_combined['date'].isin(df_test['date'])].copy()
    
    print(f"✓ Lag features created")
    
    # Remove NaN values
    df_train_clean = df_train_with_lags.dropna().copy()
    df_test_clean = df_test_with_lags.dropna().copy()
    
    print(f"✓ Training data shape: {df_train_clean.shape}")
    print(f"✓ Test data shape: {df_test_clean.shape}")
    
    return df_train_clean, df_test_clean

def split_data(df_train_clean):
    """Split data into train and validation sets"""
    print("\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    
    feature_cols = [col for col in df_train_clean.columns 
                    if col not in ['date', 'sales']]
    
    split_date = df_train_clean['date'].quantile(0.8)
    train_data = df_train_clean[df_train_clean['date'] <= split_date]
    val_data = df_train_clean[df_train_clean['date'] > split_date]
    
    X_train = train_data[feature_cols]
    y_train = train_data['sales']
    X_val = val_data[feature_cols]
    y_val = val_data['sales']
    
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Validation samples: {X_val.shape[0]}")
    print(f"✓ Features: {len(feature_cols)}")
    
    return X_train, y_train, X_val, y_val, feature_cols

def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{'='*60}")
    print(f"{model_name} PERFORMANCE")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    start_time = time.time()
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    elapsed_time = time.time() - start_time
    print(f"✓ Training completed in {elapsed_time:.2f} seconds")
    
    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, "XGBoost")
    
    return model, results

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    
    start_time = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(50)]
    )
    
    elapsed_time = time.time() - start_time
    print(f"✓ Training completed in {elapsed_time:.2f} seconds")
    
    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, "LightGBM")
    
    return model, results

def compare_models(results):
    """Compare model results"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_model_idx = results_df['rmse'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'model']
    print(f"\n🏆 Best Model: {best_model_name} (Lowest RMSE)")
    
    return results_df

def main():
    """Main training pipeline"""
    print("="*60)
    print("DEMAND FORECASTING TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Load data
    df_train, df_test = load_data()
    
    # Feature engineering
    df_train_clean, df_test_clean = prepare_features(df_train, df_test)
    
    # Split data
    X_train, y_train, X_val, y_val, feature_cols = split_data(df_train_clean)
    
    # Train models
    results = []
    
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    results.append(xgb_results)
    
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val)
    results.append(lgb_results)
    
    # Compare models
    results_df = compare_models(results)
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✓ Results saved to 'model_comparison_results.csv'")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()