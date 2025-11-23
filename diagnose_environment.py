"""
Diagnostic script to check Python environment and numpy/pandas compatibility
"""
import sys
import os

print("="*60)
print("Python Environment Diagnostic")
print("="*60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # First 3 entries
print()

# Check numpy
print("="*60)
print("NumPy Check")
print("="*60)
try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ NumPy location: {np.__file__}")
    print(f"✓ NumPy dtype size: {np.dtype('float64').itemsize * 8} bits")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Check pandas
print("="*60)
print("Pandas Check")
print("="*60)
try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
    print(f"✓ Pandas location: {pd.__file__}")
    
    # Test creating a DataFrame
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"✓ DataFrame creation successful: {df.shape}")
    
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print(f"✗ BINARY INCOMPATIBILITY DETECTED!")
        print(f"  Error: {e}")
        print()
        print("SOLUTION:")
        print("  1. Run: FIX_NUMPY_PANDAS.bat")
        print("  2. Or manually:")
        print("     pip uninstall -y numpy pandas")
        print("     pip install numpy==1.26.4")
        print("     pip install pandas==2.1.4")
        print("  3. Restart your IDE/terminal")
    else:
        print(f"✗ Pandas error: {e}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"✗ Pandas import failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Check for multiple installations
print("="*60)
print("Checking for Multiple Installations")
print("="*60)
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "show", "numpy", "pandas"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("Could not check pip installations")

print()
print("="*60)
print("If you still see errors:")
print("1. Make sure you're using the correct Python interpreter")
print("2. Restart your IDE/terminal after fixing")
print("3. Try: python -m pip install --upgrade --force-reinstall numpy==1.26.4 pandas==2.1.4")
print("="*60)

