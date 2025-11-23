@echo off
REM Script to run the Streamlit application on Windows

echo Starting Sales Forecasting Application...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create models directory if it doesn't exist
if not exist "models" mkdir models

REM Run Streamlit app
echo Starting Streamlit app...
streamlit run app.py

