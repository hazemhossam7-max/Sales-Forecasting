"""
Script to fix numpy/pandas binary incompatibility issues
Run this script to reinstall compatible versions
"""
import subprocess
import sys

def run_command(command):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    print("="*60)
    print("Fixing NumPy/Pandas Binary Incompatibility")
    print("="*60)
    
    # Step 1: Uninstall numpy and pandas
    print("\nStep 1: Uninstalling numpy and pandas...")
    run_command(f"{sys.executable} -m pip uninstall -y numpy pandas")
    
    # Step 2: Install compatible numpy first
    print("\nStep 2: Installing compatible numpy version...")
    run_command(f"{sys.executable} -m pip install 'numpy>=1.24.0,<2.0.0' --no-cache-dir")
    
    # Step 3: Install compatible pandas
    print("\nStep 3: Installing compatible pandas version...")
    run_command(f"{sys.executable} -m pip install 'pandas>=2.0.0,<2.2.0' --no-cache-dir")
    
    # Step 4: Verify installation
    print("\nStep 4: Verifying installation...")
    try:
        import numpy as np
        import pandas as pd
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Pandas version: {pd.__version__}")
        print("\n✓ Installation successful! The compatibility issue should be fixed.")
        print("\nYou can now run: streamlit run app.py")
    except ImportError as e:
        print(f"\n✗ Error importing libraries: {e}")
        print("Please try running the commands manually:")
        print("  pip uninstall -y numpy pandas")
        print("  pip install 'numpy>=1.24.0,<2.0.0'")
        print("  pip install 'pandas>=2.0.0,<2.2.0'")

if __name__ == "__main__":
    main()

