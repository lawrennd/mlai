#!/usr/bin/env python3
"""
Reproduction of the coverage collection issue with mlai.mlai import.

This demonstrates the _NoValueType error that occurs when
coverage collection interferes with numpy's internal state
after importing mlai.mlai.
"""

import numpy as np
import matplotlib.pyplot as plt

def test_mlai_import_coverage_issue():
    """Test that demonstrates the coverage collection issue with mlai import."""
    print("Testing mlai import with coverage...")
    
    # Import mlai.mlai to trigger the issue
    import mlai.mlai as mlai_module
    print("✅ mlai.mlai imported successfully")
    
    # This is the exact error that occurs
    try:
        # Create a figure with subplots (this is what fails)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        print("✅ Subplots created successfully")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❌ Subplots failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_numpy_state_after_mlai_import():
    """Test numpy state after mlai import."""
    print("Testing numpy state after mlai import...")
    
    # Check numpy's _NoValue
    print(f"NumPy _NoValue: {np._NoValue}")
    print(f"NumPy _NoValue type: {type(np._NoValue)}")
    
    # Try to use numpy operations that might fail
    try:
        arr = np.array([1, 2, 3, 4])
        result = np.min(arr)
        print(f"NumPy min operation: {result}")
        return True
    except Exception as e:
        print(f"❌ NumPy min failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MLAI Import + Coverage Collection Issue Reproduction")
    print("=" * 60)
    
    # Test without coverage
    print("Testing without coverage...")
    test_mlai_import_coverage_issue()
    test_numpy_state_after_mlai_import()
    
    print("\n" + "=" * 60)
    print("To reproduce the issue, run:")
    print("python -m pytest test_mlai_coverage_reproduction.py --cov=mlai.mlai --cov-report=term-missing -v")
    print("=" * 60)
