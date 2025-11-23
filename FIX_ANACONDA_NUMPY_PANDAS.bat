@echo off
REM Fix numpy/pandas binary incompatibility in Anaconda environment
REM This script uses the Anaconda Python specifically

echo ============================================================
echo Fixing NumPy/Pandas in Anaconda Environment
echo ============================================================
echo.

set ANACONDA_PYTHON=C:\Users\hazem\anaconda3\python.exe

echo Step 1: Uninstalling numpy and pandas from Anaconda...
"%ANACONDA_PYTHON%" -m pip uninstall -y numpy pandas
echo.

echo Step 2: Clearing pip cache...
"%ANACONDA_PYTHON%" -m pip cache purge
echo.

echo Step 3: Installing compatible numpy (1.26.4)...
"%ANACONDA_PYTHON%" -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
echo.

echo Step 4: Installing compatible pandas (2.1.4)...
"%ANACONDA_PYTHON%" -m pip install --no-cache-dir --force-reinstall "pandas==2.1.4"
echo.

echo Step 5: Verifying installation...
"%ANACONDA_PYTHON%" -c "import numpy as np; import pandas as pd; print('NumPy:', np.__version__); print('Pandas:', pd.__version__); print('SUCCESS!')"
echo.

echo ============================================================
echo Fix complete! Now try: streamlit run app.py
echo ============================================================
pause

