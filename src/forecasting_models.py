"""
Model training and prediction module
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import os


class SalesForecastingModel:
    """Wrapper class for sales forecasting models"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize model
        
        Args:
            model_type: 'linear' or 'xgboost'
        """
        self.model_type = model_type.lower()
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: linear, xgboost")
    
    def train(self, X, y, use_cv=True, n_splits=5):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target vector
            use_cv: Whether to use cross-validation
            n_splits: Number of CV splits
        """
        if use_cv:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            mae_list, rmse_list = [], []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if self.model_type == 'xgboost':
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_val)
                mae_list.append(mean_absolute_error(y_val, y_pred))
                rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            
            self.mae = np.mean(mae_list)
            self.rmse = np.mean(rmse_list)
            print(f"{self.model_type.upper()} - MAE: {self.mae:.2f}, RMSE: {self.rmse:.2f}")
        else:
            self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def WAPE(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100
