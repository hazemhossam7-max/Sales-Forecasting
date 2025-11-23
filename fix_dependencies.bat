@echo off
REM Script to fix numpy/pandas binary incompatibility on Windows

echo ============================================================
echo Fixing NumPy/Pandas Binary Incompatibility
echo ============================================================
echo.

echo Step 1: Uninstalling numpy and pandas...
python -m pip uninstall -y numpy pandas
echo.

echo Step 2: Installing compatible numpy version...
python -m pip install "numpy>=1.24.0,<2.0.0" --no-cache-dir
echo.

echo Step 3: Installing compatible pandas version...
python -m pip install "pandas>=2.0.0,<2.2.0" --no-cache-dir
echo.

echo Step 4: Verifying installation...
python -c "import numpy as np; import pandas as pd; print('NumPy version:', np.__version__); print('Pandas version:', pd.__version__); print('Installation successful!')"
echo.

echo ============================================================
echo Fix complete! You can now run: streamlit run app.py
echo ============================================================
pause

