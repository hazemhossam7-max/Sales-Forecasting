"""
Model training and prediction module
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import pickle

# LSTM imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Prophet imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class SalesForecastingModel:
    """Wrapper class for sales forecasting models"""
    
    def __init__(self, model_type='xgboost', seq_len=8, epochs=20):
        """
        Initialize model
        
        Args:
            model_type: 'linear', 'xgboost', 'lstm', or 'prophet'
            seq_len: Sequence length for LSTM (default: 8)
            epochs: Number of epochs for LSTM training (default: 20)
        """
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.epochs = epochs
        self.scaler = None
        self.feature_cols = None
        
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
        elif self.model_type == 'lstm':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
            self.model = None  # Will be created during training
            self.scaler = StandardScaler()
        elif self.model_type == 'prophet':
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet is required. Install with: pip install prophet")
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: linear, xgboost, lstm, prophet")
    
    def train(self, X, y, use_cv=True, n_splits=5, df=None):
        """
        Train the model
        
        Args:
            X: Feature matrix (for linear, xgboost, lstm)
            y: Target vector (for linear, xgboost, lstm)
            use_cv: Whether to use cross-validation (not used for Prophet)
            n_splits: Number of CV splits
            df: Full dataframe with Date column (required for Prophet)
        """
        if self.model_type == 'prophet':
            # Prophet requires aggregated time series data
            if df is None:
                raise ValueError("DataFrame with Date column is required for Prophet model")
            
            # Prepare data for Prophet: requires columns "ds" and "y"
            prophet_df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Train Prophet
            self.model.fit(prophet_df)
            self.prophet_df = prophet_df
            
            # Calculate metrics on training data
            forecast = self.model.predict(prophet_df)
            self.mae = mean_absolute_error(prophet_df['y'], forecast['yhat'])
            self.rmse = np.sqrt(mean_squared_error(prophet_df['y'], forecast['yhat']))
            print(f"PROPHET - MAE: {self.mae:.2f}, RMSE: {self.rmse:.2f}")
            
        elif self.model_type == 'lstm':
            # LSTM requires sequence data
            if use_cv:
                # For LSTM, we'll do a simple train/test split instead of full CV
                # due to sequence requirements
                X_scaled = self.scaler.fit_transform(X)
                y_values = y.values
                
                X_seq, y_seq = self._make_sequences(X_scaled, y_values, self.seq_len)
                train_size = int(len(X_seq) * 0.8)
                X_train_seq, X_val_seq = X_seq[:train_size], X_seq[train_size:]
                y_train_seq, y_val_seq = y_seq[:train_size], y_seq[train_size:]
                
                # Build and train LSTM
                self.model = Sequential([
                    LSTM(64, activation='relu', input_shape=(self.seq_len, X_seq.shape[2])),
                    Dropout(0.2),
                    Dense(1)
                ])
                self.model.compile(optimizer='adam', loss='mse')
                self.model.fit(X_train_seq, y_train_seq, epochs=self.epochs, batch_size=32, verbose=0)
                
                # Evaluate
                y_pred = self.model.predict(X_val_seq, verbose=0).flatten()
                self.mae = mean_absolute_error(y_val_seq, y_pred)
                self.rmse = np.sqrt(mean_squared_error(y_val_seq, y_pred))
                print(f"LSTM - MAE: {self.mae:.2f}, RMSE: {self.rmse:.2f}")
            else:
                X_scaled = self.scaler.fit_transform(X)
                y_values = y.values
                X_seq, y_seq = self._make_sequences(X_scaled, y_values, self.seq_len)
                
                self.model = Sequential([
                    LSTM(64, activation='relu', input_shape=(self.seq_len, X_seq.shape[2])),
                    Dropout(0.2),
                    Dense(1)
                ])
                self.model.compile(optimizer='adam', loss='mse')
                self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=32, verbose=0)
                
        else:
            # Linear Regression and XGBoost
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
    
    def predict(self, X=None, periods=12, freq='W'):
        """
        Make predictions
        
        Args:
            X: Feature matrix (for linear, xgboost, lstm)
            periods: Number of future periods to forecast (for Prophet)
            freq: Frequency string for Prophet (default: 'W' for weekly)
        
        Returns:
            Predictions array or DataFrame
        """
        if self.model_type == 'prophet':
            # Prophet forecasting
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        elif self.model_type == 'lstm':
            # LSTM requires sequence data
            if X is None:
                raise ValueError("X is required for LSTM predictions")
            # Convert to numpy array if DataFrame
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            X_scaled = self.scaler.transform(X_array)
            # For prediction, we need to create sequences from the last seq_len rows
            if len(X_scaled) < self.seq_len:
                raise ValueError(f"Need at least {self.seq_len} samples for LSTM prediction")
            
            # Use last seq_len samples to make prediction
            X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, X_scaled.shape[1])
            prediction = self.model.predict(X_seq, verbose=0)
            return prediction.flatten()
        
        else:
            # Linear Regression and XGBoost
            if X is None:
                raise ValueError("X is required for predictions")
            return self.model.predict(X)
    
    def _make_sequences(self, X, y, seq_len):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    def save(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'lstm':
            # Save Keras model and scaler separately
            model_path = filepath.replace('.pkl', '_model.h5')
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            self.model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"LSTM model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
        elif self.model_type == 'prophet':
            # Save Prophet model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Prophet model saved to {filepath}")
        else:
            # Save sklearn/xgboost models
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        if self.model_type == 'lstm':
            # Load Keras model and scaler
            model_path = filepath.replace('.pkl', '_model.h5')
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required to load LSTM model")
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"LSTM model loaded from {model_path}")
        elif self.model_type == 'prophet':
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Prophet model loaded from {filepath}")
        else:
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

