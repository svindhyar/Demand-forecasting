"""
Store Item Demand Forecasting - Advanced Training Script
With command-line arguments for hyperparameter tuning
"""

import argparse
import pandas as pd
import numpy as np
import warnings
import time
import json
from datetime import datetime

# Machine Learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train demand forecasting models')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='data/train.csv',
                       help='Path to training data')
    parser.add_argument('--test_path', type=str, default='data/test.csv',
                       help='Path to test data')
    
    # Model selection
    parser.add_argument('--model', type=str, default='both',
                       choices=['xgboost', 'lightgbm', 'both'],
                       help='Model to train')
    
    # XGBoost hyperparameters
    parser.add_argument('--xgb_n_estimators', type=int, default=200,
                       help='XGBoost number of estimators')
    parser.add_argument('--xgb_max_depth', type=int, default=8,
                       help='XGBoost max depth')
    parser.add_argument('--xgb_lr', type=float, default=0.1,
                       help='XGBoost learning rate')
    
    # LightGBM hyperparameters
    parser.add_argument('--lgb_n_estimators', type=int, default=200,
                       help='LightGBM number of estimators')
    parser.add_argument('--lgb_max_depth', type=int, default=8,
                       help='LightGBM max depth')
    parser.add_argument('--lgb_lr', type=float, default=0.1,
                       help='LightGBM learning rate')
    
    # Training arguments
    parser.add_argument('--val_split', type=float, default=0.8,
                       help='Train/validation split ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', type=int, default=50,
                       help='Verbosity level')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    return parser.parse_args()

def load_data(train_path, test_path):
    """Load and preprocess data"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    print(f"✓ Training data loaded: {df_train.shape}")
    print(f"✓ Test data loaded: {df_test.shape}")
    
    return df_train, df_test

def create_features(df):
    """Create comprehensive features"""
    df = df.copy()
    
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
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def create_lag_features(df, lags=[7, 14, 28, 30, 60, 90]):
    """Create lag features"""
    df = df.copy()
    df = df.sort_values(['store', 'item', 'date'])
    
    if 'sales' in df.columns:
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
        
        for window in [7, 14, 30, 60]:
            df[f'rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
        
        df['ewm_7'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).ewm(span=7, adjust=False).mean()
        )
        df['ewm_30'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(1).ewm(span=30, adjust=False).mean()
        )
    
    return df

def prepare_features(df_train, df_test):
    """Prepare features"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    df_train = create_features(df_train)
    df_test = create_features(df_test)
    df_train_with_lags = create_lag_features(df_train)
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    df_combined = create_lag_features(df_combined)
    df_test_with_lags = df_combined[df_combined['date'].isin(df_test['date'])].copy()
    
    df_train_clean = df_train_with_lags.dropna().copy()
    df_test_clean = df_test_with_lags.dropna().copy()
    
    print(f"✓ Training data shape: {df_train_clean.shape}")
    print(f"✓ Test data shape: {df_test_clean.shape}")
    
    return df_train_clean, df_test_clean

def split_data(df_train_clean, val_split):
    """Split data"""
    print("\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    
    feature_cols = [col for col in df_train_clean.columns 
                    if col not in ['date', 'sales']]
    
    split_date = df_train_clean['date'].quantile(val_split)
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
    """Evaluate model"""
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

def train_xgboost(X_train, y_train, X_val, y_val, args):
    """Train XGBoost"""
    print("\n" + "="*60)
    print(f"TRAINING XGBOOST (lr={args.xgb_lr}, depth={args.xgb_max_depth}, n_est={args.xgb_n_estimators})")
    print("="*60)
    
    start_time = time.time()
    
    model = xgb.XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_seed,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=args.verbose)
    
    elapsed = time.time() - start_time
    print(f"✓ Training completed in {elapsed:.2f} seconds")
    
    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, "XGBoost")
    
    return model, results

def train_lightgbm(X_train, y_train, X_val, y_val, args):
    """Train LightGBM"""
    print("\n" + "="*60)
    print(f"TRAINING LIGHTGBM (lr={args.lgb_lr}, depth={args.lgb_max_depth}, n_est={args.lgb_n_estimators})")
    print("="*60)
    
    start_time = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=args.lgb_n_estimators,
        max_depth=args.lgb_max_depth,
        learning_rate=args.lgb_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_seed,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.log_evaluation(args.verbose)])
    
    elapsed = time.time() - start_time
    print(f"✓ Training completed in {elapsed:.2f} seconds")
    
    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, "LightGBM")
    
    return model, results

def main():
    """Main pipeline"""
    args = parse_args()
    
    print("="*60)
    print("DEMAND FORECASTING TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Train path: {args.train_path}")
    print(f"  Test path: {args.test_path}")
    print(f"  Val split: {args.val_split}")
    print(f"  Random seed: {args.random_seed}")
    
    df_train, df_test = load_data(args.train_path, args.test_path)
    df_train_clean, df_test_clean = prepare_features(df_train, df_test)
    X_train, y_train, X_val, y_val, feature_cols = split_data(df_train_clean, args.val_split)
    
    results = []
    models = {}
    
    if args.model in ['xgboost', 'both']:
        xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val, args)
        results.append(xgb_results)
        models['xgboost'] = xgb_model
    
    if args.model in ['lightgbm', 'both']:
        lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val, args)
        results.append(lgb_results)
        models['lightgbm'] = lgb_model
    
    # Save results
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'{args.output_dir}/results_{timestamp}.csv', index=False)
    print(f"\n✓ Results saved to {args.output_dir}/results_{timestamp}.csv")
    
    # Save models
    if args.save_models:
        import pickle
        for name, model in models.items():
            model_path = f'{args.output_dir}/{name}_model_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ {name} model saved to {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
