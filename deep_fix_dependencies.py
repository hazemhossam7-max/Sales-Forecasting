"""
Deep fix for numpy/pandas binary incompatibility
This script clears cache and forces a complete reinstall
"""
import subprocess
import sys
import os
import shutil

def run_command(command, check=True):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr)
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
    return result.returncode == 0

def clear_python_cache():
    """Clear Python cache files"""
    print("\n" + "="*60)
    print("Step 0: Clearing Python cache...")
    print("="*60)
    
    # Find __pycache__ directories
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            cache_dirs.append(cache_path)
    
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"Removed: {cache_dir}")
        except Exception as e:
            print(f"Could not remove {cache_dir}: {e}")
    
    # Clear .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                    print(f"Removed: {os.path.join(root, file)}")
                except Exception as e:
                    pass

def main():
    print("="*60)
    print("DEEP FIX: NumPy/Pandas Binary Incompatibility")
    print("="*60)
    
    # Step 0: Clear cache
    clear_python_cache()
    
    # Step 1: Uninstall everything
    print("\n" + "="*60)
    print("Step 1: Uninstalling numpy, pandas, and related packages...")
    print("="*60)
    run_command(f"{sys.executable} -m pip uninstall -y numpy pandas", check=False)
    
    # Step 2: Clear pip cache
    print("\n" + "="*60)
    print("Step 2: Clearing pip cache...")
    print("="*60)
    run_command(f"{sys.executable} -m pip cache purge", check=False)
    
    # Step 3: Install numpy first (without dependencies)
    print("\n" + "="*60)
    print("Step 3: Installing compatible numpy (fresh install)...")
    print("="*60)
    run_command(f"{sys.executable} -m pip install --no-cache-dir --force-reinstall --no-deps 'numpy==1.26.4'")
    
    # Step 4: Install numpy dependencies
    print("\n" + "="*60)
    print("Step 4: Installing numpy dependencies...")
    print("="*60)
    run_command(f"{sys.executable} -m pip install --no-cache-dir 'numpy==1.26.4'")
    
    # Step 5: Install pandas (fresh install)
    print("\n" + "="*60)
    print("Step 5: Installing compatible pandas (fresh install)...")
    print("="*60)
    run_command(f"{sys.executable} -m pip install --no-cache-dir --force-reinstall 'pandas==2.1.4'")
    
    # Step 6: Verify
    print("\n" + "="*60)
    print("Step 6: Verifying installation...")
    print("="*60)
    try:
        # Clear any cached imports
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        if 'pandas' in sys.modules:
            del sys.modules['pandas']
        
        import numpy as np
        import pandas as pd
        
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Pandas version: {pd.__version__}")
        print(f"✓ NumPy location: {np.__file__}")
        print(f"✓ Pandas location: {pd.__file__}")
        
        # Test actual import
        df = pd.DataFrame({'test': [1, 2, 3]})
        print(f"✓ Test DataFrame created: {df.shape}")
        
        print("\n" + "="*60)
        print("✓ SUCCESS! Installation verified.")
        print("="*60)
        print("\nYou can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTry running the commands manually:")
        print("  pip uninstall -y numpy pandas")
        print("  pip cache purge")
        print("  pip install --no-cache-dir --force-reinstall numpy==1.26.4")
        print("  pip install --no-cache-dir --force-reinstall pandas==2.1.4")

if __name__ == "__main__":
    main()

