"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path, features_path, stores_path, test_path=None):
    """
    Load and merge Walmart sales data
    
    Args:
        train_path: Path to train.csv
        features_path: Path to features.csv
        stores_path: Path to stores.csv
        test_path: Optional path to test.csv
    
    Returns:
        Merged dataframe
    """
    train = pd.read_csv(train_path, parse_dates=['Date'])
    features = pd.read_csv(features_path, parse_dates=['Date'])
    stores = pd.read_csv(stores_path)
    
    # Merge datasets
    df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    
    if test_path:
        test = pd.read_csv(test_path, parse_dates=['Date'])
        return df, test
    
    return df


def feature_engineering(df):
    """
    Create features for the model
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with engineered features
    """
    # Date-based features
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Lag features for Weekly_Sales
    df = df.sort_values(['Store', 'Dept', 'Date'])
    df['Sales_Lag_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    df['Sales_Lag_2'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
    df['Sales_Lag_3'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(3)
    
    # Rolling mean
    df['Rolling_Mean_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=4).mean()
    
    # Handle missing markdowns
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Encode store type
    le_type = LabelEncoder()
    df['Type_encoded'] = le_type.fit_transform(df['Type'])
    
    # Drop or fill any remaining NaNs
    df.fillna(0, inplace=True)
    
    return df


def prepare_features(df):
    """
    Prepare feature columns and target for modeling
    
    Args:
        df: Dataframe with engineered features
    
    Returns:
        X (features), y (target), feature_cols (list of feature names)
    """
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    feature_cols = [
        'Store', 'Dept', 'Year', 'Week', 'Month', 'DayOfWeek', 'IsHoliday',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    ] + markdown_cols + [
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Rolling_Mean_4', 'Type_encoded'
    ]
    
    target = 'Weekly_Sales'
    
    # Drop rows with no lag (first few weeks)
    df_model = df.dropna(subset=['Sales_Lag_3', 'Rolling_Mean_4'])
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    return X, y, feature_cols

