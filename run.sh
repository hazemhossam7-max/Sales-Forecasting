#!/bin/bash
# Script to run the Streamlit application

echo "Starting Sales Forecasting Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

# Run Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py

