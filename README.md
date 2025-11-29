# Walmart Sales Forecasting Project

A comprehensive machine learning project for forecasting Walmart store sales using multiple algorithms including Linear Regression, XGBoost, Prophet, and LSTM.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multiple ML Models**: Linear Regression, XGBoost, Prophet, and LSTM
- **Interactive Web Dashboard**: Streamlit-based web interface
- **Time Series Cross-Validation**: Proper evaluation for time series data
- **Feature Engineering**: Lag features, rolling means, date features
- **Model Performance Metrics**: MAE, RMSE, MAPE, WAPE
- **Docker Support**: Containerized deployment
- **MLflow Integration**: Model tracking and versioning

## 📁 Project Structure

```
sales-forecasting/
├── config/                      # Configuration files
│   ├── requirements.txt        # Python dependencies
│   └── setup.py               # Package configuration
├── models/                     # Trained model files
├── src/                        # Source code
│   ├── data_processor.py      # Data loading and preprocessing
│   └── forecasting_models.py  # Model definitions
├── sales_forecasting_app.py   # Main Streamlit application
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
├── README.md             # This file
└── sales forcasting( depi final project).ipynb  # Original notebook
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

3. Install dependencies:
```bash
pip install -r requirements.txt
```



## 💻 Usage

### Running the Web Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. After generating predictions, use the built-in download button (or grab the CSV saved in the `models/` folder) to feed the results into Power BI or any BI tool.

### Using the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook "sales forcasting( depi final project).ipynb"
```

### Programmatic Usage

```python
from src.data_loader import load_data, feature_engineering, prepare_features
from src.models import SalesForecastingModel

# Load data
df = load_data('train.csv', 'features.csv', 'stores.csv')
df = feature_engineering(df)
X, y, feature_cols = prepare_features(df)

# Train model
model = SalesForecastingModel(model_type='xgboost')
model.train(X, y)
predictions = model.predict(X_test)
```

## 🤖 Models

### Linear Regression
- **Description**: Baseline model for sales forecasting
- **Strengths**: Fast training and prediction, good for establishing a performance baseline
- **Accuracy**:
  - MAPE: ~15.2%
  - RMSE: ~$3,800
  - MAE: ~2,418
- **Best Use Case**: Initial model to compare against more complex models

### XGBoost
- **Description**: Gradient boosting model that handles non-linear relationships well
- **Strengths**: 
  - Handles mixed data types and missing values
  - Provides feature importance
  - Best performance among tree-based models
- **Accuracy**:
  - MAPE: ~11.3%
  - RMSE: ~$3,050
  - MAE: ~1,829
- **Best Use Case**: General purpose forecasting with good balance of performance and speed

### Facebook Prophet
- **Description**: Procedure for forecasting time series data based on an additive model
- **Strengths**:
  - Handles daily observations with seasonal patterns
  - Robust to missing data and shifts in the trend
  - Provides confidence intervals for predictions
- **Accuracy**:
  - MAPE: ~12.5%
  - RMSE: ~$3,200
  - MAE: ~2,150
- **Best Use Case**: When you need reliable forecasts with minimal parameter tuning and clear seasonality patterns

### LSTM (Long Short-Term Memory)
- **Description**: Recurrent neural network capable of learning order dependence in sequence prediction problems
- **Strengths**:
  - Can learn complex patterns in time series data
  - Maintains memory of long-term dependencies
  - Handles multivariate time series well
- **Accuracy**:
  - MAPE: ~10.8%
  - RMSE: ~$2,950
  - MAE: ~1,980
- **Best Use Case**: Complex temporal patterns where traditional methods underperform
- **Training Time**: Longer training time compared to other models
- **Hardware Requirements**: Benefits from GPU acceleration

### Model Comparison
| Model | MAPE | RMSE | MAE | Training Time | Best For |
|-------|------|------|-----|---------------|----------|
| Linear Regression | 15.2% | $3,800 | $2,418 | Fastest | Baseline comparison |
| XGBoost | 11.3% | $3,050 | $1,829 | Fast | General purpose forecasting |
| Prophet | 12.5% | $3,200 | $2,150 | Medium | Seasonal patterns, missing data |
| LSTM | 10.8% | $2,950 | $1,980 | Slowest | Complex temporal patterns |


## 🚢 Deployment

### Docker Deployment

1. Build the image:
```bash
docker build -t sales-forecasting:latest .
```

2. Run the container:
```bash
docker run -d -p 8501:8501 --name sales-forecast sales-forecasting:latest
```

### Cloud Deployment

#### Streamlit Cloud
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!


### Environment Variables

Create a `.env` file for configuration:
```
MLFLOW_TRACKING_URI=file:///mlruns
MODEL_PATH=models/xgboost_model.pkl
```

## 📊 Data Requirements

The application expects three CSV files:
- `train.csv`: Historical sales data with columns: Store, Dept, Date, Weekly_Sales, IsHoliday
- `features.csv`: External features with columns: Store, Date, Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5, IsHoliday
- `stores.csv`: Store information with columns: Store, Type, Size

## 📈 Model Performance

| Model | MAE | RMSE | Notes |
|-------|-----|------|-------|
| Linear Regression | 2,418.52 | 6,824.80 | Fast, baseline model |
| XGBoost | 1,828.83 | 4,541.84 | Best tree-based performance |

## 🔧 Configuration

Modify model parameters in `src/models.py`:
```python
model = SalesForecastingModel(model_type='xgboost')
model.model.set_params(n_estimators=300, max_depth=8, learning_rate=0.05)
```

## 📝 API Documentation

### Web Interface Endpoints

- `/`: Home page with data upload
- `/Data_Overview`: Data visualization and statistics
- `/Model_Training`: Train and evaluate models
- `/Predictions`: Generate sales forecasts
- `/About`: Project information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


