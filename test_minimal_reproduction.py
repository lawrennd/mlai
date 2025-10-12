#!/usr/bin/env python3
"""
Minimal reproduction of the matplotlib/numpy compatibility issue.

This script demonstrates the _NoValueType error that occurs when
matplotlib tries to create subplots in certain conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy

def test_matplotlib_issue():
    """Test that reproduces the matplotlib _NoValueType error."""
    print("Testing matplotlib subplots...")
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        print("✅ Basic subplots: OK")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Basic subplots failed: {e}")
        print(f"Error type: {type(e)}")
        return False
    return True

def test_scipy_issue():
    """Test that reproduces the scipy VoidDType error."""
    print("Testing scipy linkage...")
    try:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = scipy.cluster.hierarchy.linkage(X, method='ward')
        print("✅ SciPy linkage: OK")
        return True
    except Exception as e:
        print(f"❌ SciPy linkage failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_combined_issue():
    """Test that reproduces the combined issue."""
    print("Testing combined matplotlib + scipy...")
    try:
        # First create a matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        
        # Then try scipy
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = scipy.cluster.hierarchy.linkage(X, method='ward')
        
        print("✅ Combined test: OK")
        return True
    except Exception as e:
        print(f"❌ Combined test failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_numpy_state():
    """Test numpy state after matplotlib operations."""
    print("Testing numpy state after matplotlib...")
    try:
        # Create and close a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        
        # Check numpy state
        print(f"NumPy _NoValue: {np._NoValue}")
        print(f"NumPy _NoValue type: {type(np._NoValue)}")
        print(f"NumPy _NoValue value: {np._NoValue}")
        
        # Try to use numpy operations
        arr = np.array([1, 2, 3, 4])
        result = np.min(arr)
        print(f"NumPy min operation: {result}")
        
        print("✅ NumPy state: OK")
        return True
    except Exception as e:
        print(f"❌ NumPy state failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_matplotlib_backend():
    """Test matplotlib backend state."""
    print("Testing matplotlib backend...")
    try:
        print(f"Current backend: {plt.get_backend()}")
        
        # Try to create a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        
        print("✅ Matplotlib backend: OK")
        return True
    except Exception as e:
        print(f"❌ Matplotlib backend failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def main():
    """Run all tests to identify the issue."""
    print("=" * 60)
    print("Minimal Reproduction of Matplotlib/NumPy/SciPy Issues")
    print("=" * 60)
    
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    print()
    
    tests = [
        test_matplotlib_issue,
        test_scipy_issue,
        test_combined_issue,
        test_numpy_state,
        test_matplotlib_backend
    ]
    
    results = []
    for test in tests:
        print("-" * 40)
        result = test()
        results.append(result)
        print()
    
    print("=" * 60)
    print("Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if not all(results):
        print("\n❌ Issues detected! This reproduces the test failures.")
    else:
        print("\n✅ All tests passed. No issues detected.")

if __name__ == "__main__":
    main()
