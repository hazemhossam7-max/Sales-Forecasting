# Models Update - LSTM and Prophet Added

## ✅ Changes Made

### 1. Updated `src/models.py`
- **Added LSTM Model Support**:
  - Sequence-based data preparation
  - Configurable sequence length and epochs
  - StandardScaler for feature normalization
  - Keras/TensorFlow integration
  - Model saving/loading for .h5 files and scalers

- **Added Prophet Model Support**:
  - Time series aggregation (grouped by Date)
  - Automatic seasonality detection
  - Confidence intervals in predictions
  - Prophet-specific save/load functionality

- **Enhanced SalesForecastingModel Class**:
  - Now supports 4 model types: `linear`, `xgboost`, `lstm`, `prophet`
  - Model-specific training methods
  - Model-specific prediction methods
  - Graceful handling of missing dependencies

### 2. Updated `app.py`
- **Model Selection**:
  - Added "LSTM" and "Prophet" to model dropdown
  - Added LSTM-specific parameters (sequence length, epochs) with sliders
  - Different training flows for each model type

- **Enhanced Predictions Page**:
  - **Prophet**: Shows aggregated sales forecast with confidence intervals
  - **LSTM**: Uses sequence-based predictions for store/dept combinations
  - **Linear/XGBoost**: Improved prediction interface
  - Better error handling and user feedback

- **Updated Home Page**:
  - Updated feature list to include all 4 models

### 3. Updated Documentation
- **README.md**: 
  - Added detailed descriptions for LSTM and Prophet
  - Updated model comparison table
  - Added usage examples for all models

- **.gitignore**: 
  - Added LSTM model files (*.h5, *_scaler.pkl)
  - Ensures model files aren't committed

## 🎯 Model Features

### LSTM Model
- **Input**: Sequence of features (configurable length, default: 8 weeks)
- **Architecture**: 64-unit LSTM layer with 0.2 dropout + Dense output
- **Training**: Configurable epochs (default: 20)
- **Use Case**: Best for capturing complex temporal patterns
- **Requirements**: TensorFlow

### Prophet Model
- **Input**: Aggregated time series (Date, Weekly_Sales)
- **Features**: Automatic seasonality, holiday effects, confidence intervals
- **Output**: Forecast with upper/lower bounds
- **Use Case**: Best for overall sales trends and seasonality
- **Requirements**: Prophet library

## 📊 Usage Examples

### Training LSTM
```python
from src.models import SalesForecastingModel
from src.data_loader import load_data, feature_engineering, prepare_features

# Load and prepare data
df = load_data('train.csv', 'features.csv', 'stores.csv')
df = feature_engineering(df)
X, y, _ = prepare_features(df)

# Train LSTM
model = SalesForecastingModel(model_type='lstm', seq_len=8, epochs=20)
model.train(X, y, use_cv=True)
predictions = model.predict(X_test)
```

### Training Prophet
```python
from src.models import SalesForecastingModel
from src.data_loader import load_data, feature_engineering

# Load data (Prophet needs full dataframe with Date)
df = load_data('train.csv', 'features.csv', 'stores.csv')
df = feature_engineering(df)

# Train Prophet
model = SalesForecastingModel(model_type='prophet')
model.train(X=None, y=None, df=df)
forecast = model.predict(periods=12, freq='W')
```

## 🚀 Deployment Notes

1. **Dependencies**: All required packages are in `requirements.txt`
   - TensorFlow for LSTM
   - Prophet for time series forecasting

2. **Model Files**:
   - LSTM saves: `*_model.h5` and `*_scaler.pkl`
   - Prophet saves: `*.pkl`
   - All model files are gitignored

3. **Performance**:
   - LSTM training can take longer (depends on epochs and data size)
   - Prophet is generally faster for aggregated data
   - Both models are production-ready

## 🔄 Next Steps

To use the new models:
1. Run the Streamlit app: `streamlit run app.py`
2. Load your data from the Home page
3. Go to Model Training and select LSTM or Prophet
4. Adjust parameters if needed
5. Train the model
6. Use the Predictions page to generate forecasts

All changes have been committed and are ready to push to GitHub!

