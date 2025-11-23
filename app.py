"""
Streamlit web application for Sales Forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, feature_engineering, prepare_features
from src.models import SalesForecastingModel, MAPE, WAPE

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Walmart Sales Forecasting Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Model Training", "Predictions", "About"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Home Page
if page == "Home":
    st.header("Welcome to Sales Forecasting Dashboard")
    st.markdown("""
    This application provides:
    - **Data Overview**: Explore Walmart sales data
    - **Model Training**: Train forecasting models (Linear Regression, XGBoost)
    - **Predictions**: Make sales forecasts for future dates
    
    ### Features
    - Time series forecasting using multiple ML models
    - Interactive data visualization
    - Model performance metrics
    - Future sales predictions
    """)
    
    # File uploader for data
    st.subheader("📁 Load Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_file = st.file_uploader("Upload train.csv", type=['csv'])
    with col2:
        features_file = st.file_uploader("Upload features.csv", type=['csv'])
    with col3:
        stores_file = st.file_uploader("Upload stores.csv", type=['csv'])
    
    if st.button("Load Data") and train_file and features_file and stores_file:
        try:
            with st.spinner("Loading data..."):
                # Save uploaded files temporarily
                train_path = "temp_train.csv"
                features_path = "temp_features.csv"
                stores_path = "temp_stores.csv"
                
                with open(train_path, "wb") as f:
                    f.write(train_file.getbuffer())
                with open(features_path, "wb") as f:
                    f.write(features_file.getbuffer())
                with open(stores_path, "wb") as f:
                    f.write(stores_file.getbuffer())
                
                # Load and process data
                df = load_data(train_path, features_path, stores_path)
                df = feature_engineering(df)
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Clean up temp files
                os.remove(train_path)
                os.remove(features_path)
                os.remove(stores_path)
            
            st.success("✅ Data loaded successfully!")
            st.info(f"Dataset shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Data Overview Page
elif page == "Data Overview":
    st.header("📊 Data Overview")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
    else:
        df = st.session_state.df
        
        # Basic statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Stores", df['Store'].nunique())
        with col3:
            st.metric("Departments", df['Dept'].nunique())
        with col4:
            st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Total sales over time
        fig, ax = plt.subplots(figsize=(12, 6))
        sales_sum = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        sns.lineplot(data=sales_sum, x='Date', y='Weekly_Sales', ax=ax)
        ax.set_title("Total Weekly Sales Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Weekly Sales")
        st.pyplot(fig)
        
        # Sales by store type
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Type', y='Weekly_Sales', ax=ax)
        ax.set_title("Weekly Sales by Store Type")
        st.pyplot(fig)
        
        # Correlation heatmap
        numeric_cols = ['Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        available_cols = [col for col in numeric_cols if col in df.columns]
        if available_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[available_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(100))

# Model Training Page
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first.")
    else:
        df = st.session_state.df
        
        st.subheader("Train Model")
        model_type = st.selectbox("Select Model", ["XGBoost", "Linear Regression"])
        
        if st.button("Train Model"):
            try:
                with st.spinner("Training model..."):
                    X, y, feature_cols = prepare_features(df)
                    
                    model = SalesForecastingModel(model_type=model_type.lower().replace(" ", "_"))
                    model.train(X, y, use_cv=True, n_splits=5)
                    
                    st.session_state.model = model
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.feature_cols = feature_cols
                    st.session_state.model_type = model_type.lower().replace(" ", "_")
                
                st.success("✅ Model trained successfully!")
                if hasattr(model, 'mae') and hasattr(model, 'rmse'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MAE", f"${model.mae:,.2f}")
                    with col2:
                        st.metric("RMSE", f"${model.rmse:,.2f}")
                
                # Save model
                model_path = f"models/{model_type_lower}_model.pkl"
                model.save(model_path)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Model comparison if both models exist
        if st.session_state.model:
            st.subheader("Model Performance")
            st.info(f"Current model: {model_type}")
            if hasattr(st.session_state.model, 'mae'):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"${st.session_state.model.mae:,.2f}")
                with col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"${st.session_state.model.rmse:,.2f}")

# Predictions Page
elif page == "Predictions":
    st.header("🔮 Sales Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model from the Model Training page first.")
    else:
        st.subheader("Make Predictions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            store = st.number_input("Store ID", min_value=1, max_value=45, value=1)
        with col2:
            dept = st.number_input("Department ID", min_value=1, max_value=99, value=1)
        with col3:
            weeks_ahead = st.number_input("Weeks to Forecast", min_value=1, max_value=52, value=4)
        
        if st.button("Generate Forecast"):
            try:
                df = st.session_state.df
                model = st.session_state.model
                feature_cols = st.session_state.feature_cols
                
                # Get last known data for this store/dept
                store_dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
                
                if len(store_dept_data) == 0:
                    st.error("No historical data found for this Store/Department combination.")
                else:
                    # Prepare future dates
                    last_date = store_dept_data['Date'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=weeks_ahead, freq='W')
                    
                    # Create predictions (simplified - would need proper feature engineering for future dates)
                    st.info("⚠️ Note: This is a simplified prediction. For production, you would need to properly engineer features for future dates.")
                    
                    # Get average historical sales for this store/dept
                    avg_sales = store_dept_data['Weekly_Sales'].mean()
                    
                    # Simple forecast (in production, use proper feature engineering)
                    predictions = [avg_sales * (1 + np.random.normal(0, 0.1)) for _ in range(weeks_ahead)]
                    
                    # Display results
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Sales': predictions
                    })
                    
                    st.subheader("Forecast Results")
                    st.dataframe(forecast_df)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(12, 6))
                    historical = store_dept_data.groupby('Date')['Weekly_Sales'].sum().reset_index()
                    sns.lineplot(data=historical, x='Date', y='Weekly_Sales', label='Historical', ax=ax)
                    sns.lineplot(data=forecast_df, x='Date', y='Predicted_Sales', label='Forecast', ax=ax, marker='o')
                    ax.set_title(f"Sales Forecast for Store {store}, Department {dept}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Weekly Sales")
                    ax.legend()
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

# About Page
elif page == "About":
    st.header("ℹ️ About")
    st.markdown("""
    ## Sales Forecasting Project
    
    This application implements multiple machine learning models for forecasting Walmart sales:
    
    ### Models Implemented
    - **Linear Regression**: Baseline model for sales forecasting
    - **XGBoost**: Gradient boosting model for improved accuracy
    
    ### Features
    - Time series cross-validation
    - Feature engineering (lag features, rolling means, date features)
    - Model performance metrics (MAE, RMSE, MAPE, WAPE)
    - Interactive web interface
    
    ### Technology Stack
    - Python 3.10+
    - Streamlit for web interface
    - Scikit-learn, XGBoost, TensorFlow for ML models
    - Pandas, NumPy for data processing
    - Matplotlib, Seaborn for visualization
    
    ### Deployment
    This application can be deployed using:
    - Docker containers
    - Cloud platforms (AWS, GCP, Azure)
    - Streamlit Cloud
    - Heroku
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

