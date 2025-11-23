# Troubleshooting Guide

## NumPy/Pandas Binary Incompatibility Error

### Error Message
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

### Cause
This error occurs when there's a version mismatch between NumPy and Pandas. Specifically:
- NumPy 2.x is incompatible with Pandas 2.0.x - 2.1.x
- Pandas needs to be recompiled against the NumPy version you're using

### Solution

**For Anaconda Users (Most Common):**
```bash
FIX_ANACONDA_NUMPY_PANDAS.bat
```

Or manually:
```bash
C:\Users\hazem\anaconda3\python.exe -m pip uninstall -y numpy pandas
C:\Users\hazem\anaconda3\python.exe -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
C:\Users\hazem\anaconda3\python.exe -m pip install --no-cache-dir --force-reinstall "pandas==2.1.4"
```

**For System Python Users:**
```bash
FIX_NUMPY_PANDAS.bat
```

**Option 2: Use the fix script (Recommended)**
```bash
# Windows
fix_dependencies.bat

# Linux/Mac
chmod +x fix_dependencies.sh
./fix_dependencies.sh

# Or Python script
python fix_dependencies.py
```

**Option 2: Manual fix**
```bash
# Uninstall incompatible versions
pip uninstall -y numpy pandas

# Install compatible versions (in this order)
pip install "numpy>=1.24.0,<2.0.0"
pip install "pandas>=2.0.0,<2.2.0"
```

**Option 3: Use a virtual environment (Best Practice)**
```bash
# Create fresh virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Compatible Versions

| Component | Compatible Versions |
|-----------|---------------------|
| NumPy | 1.24.0 - 1.26.x (NOT 2.x) |
| Pandas | 2.0.0 - 2.1.x |
| Python | 3.10+ |

### Verification

After fixing, verify the installation:
```python
import numpy as np
import pandas as pd
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

### Common Issues

1. **OpenCV conflicts**: If you have OpenCV installed, it may require NumPy 2.x. Consider using a separate environment for this project.

2. **Anaconda environments**: If using Anaconda, you may need to:
   ```bash
   conda install numpy=1.26.4 pandas=2.1.4
   ```

3. **Multiple Python installations**: Make sure you're using the correct Python interpreter. Check with:
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   ```

### Additional Solutions

**If the error persists after fixing:**

1. **Restart your IDE/terminal** - Python caches imports, restart to clear them

2. **Check you're using the correct Python**:
   ```bash
   python --version
   which python  # Linux/Mac
   where python  # Windows
   ```

3. **Run diagnostic script**:
   ```bash
   python diagnose_environment.py
   ```

4. **Force reinstall with specific versions**:
   ```bash
   python -m pip install --upgrade --force-reinstall --no-cache-dir numpy==1.26.4 pandas==2.1.4
   ```

5. **Use a virtual environment** (Recommended for isolation):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

### Prevention

To avoid this issue in the future:
1. Always use `requirements.txt` with version constraints
2. Use virtual environments for each project
3. Install NumPy before Pandas when setting up new environments
4. Pin specific versions in production deployments

