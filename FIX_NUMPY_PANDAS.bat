@echo off
REM Simple fix for numpy/pandas binary incompatibility
REM Run this as Administrator if needed

echo ============================================================
echo Fixing NumPy/Pandas Binary Incompatibility
echo ============================================================
echo.

echo Step 1: Uninstalling numpy and pandas...
python -m pip uninstall -y numpy pandas
echo.

echo Step 2: Clearing pip cache...
python -m pip cache purge
echo.

echo Step 3: Installing compatible numpy...
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
echo.

echo Step 4: Installing compatible pandas...
python -m pip install --no-cache-dir --force-reinstall "pandas==2.1.4"
echo.

echo Step 5: Verifying...
python -c "import numpy as np; import pandas as pd; print('NumPy:', np.__version__); print('Pandas:', pd.__version__); print('SUCCESS!')"
echo.

echo ============================================================
echo Fix complete! Try running: streamlit run app.py
echo ============================================================
pause

