#!/usr/bin/env python3
"""
Reproduction of the matplotlib/numpy compatibility issue in test environment.

This script simulates the test environment to reproduce the _NoValueType error.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import sys
import os

def simulate_test_environment():
    """Simulate the test environment that causes the issue."""
    print("Simulating test environment...")
    
    # Simulate pytest environment
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    print()
    
    # Test 1: Basic matplotlib operations
    print("Test 1: Basic matplotlib operations")
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Basic matplotlib: OK")
    except Exception as e:
        print(f"❌ Basic matplotlib failed: {e}")
        return False
    
    # Test 2: SciPy operations
    print("Test 2: SciPy operations")
    try:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = scipy.cluster.hierarchy.linkage(X, method='ward')
        print("✅ SciPy operations: OK")
    except Exception as e:
        print(f"❌ SciPy operations failed: {e}")
        return False
    
    # Test 3: Simulate test pollution
    print("Test 3: Simulating test pollution")
    try:
        # Create multiple figures without closing them
        for i in range(5):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [4, 5, 6])
            # Don't close the figure to simulate pollution
        
        # Now try to create a new figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Test pollution simulation: OK")
    except Exception as e:
        print(f"❌ Test pollution simulation failed: {e}")
        return False
    
    # Test 4: Simulate coverage collection
    print("Test 4: Simulating coverage collection")
    try:
        # Simulate what happens during coverage collection
        import importlib
        import mlai.mlai as mlai_module
        
        # Try to create a figure after importing mlai
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Coverage simulation: OK")
    except Exception as e:
        print(f"❌ Coverage simulation failed: {e}")
        return False
    
    # Test 5: Simulate the specific failing test
    print("Test 5: Simulating the specific failing test")
    try:
        # This is the exact pattern from the failing test
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai_module.WardsMethod(X)
        ward.fit()
        
        linkage_matrix = ward.get_linkage_matrix()
        
        # Try to create dendrogram
        fig, ax = plt.subplots(figsize=(6, 4))
        scipy.cluster.hierarchy.dendrogram(linkage_matrix, ax=ax)
        plt.close(fig)
        print("✅ Specific failing test: OK")
    except Exception as e:
        print(f"❌ Specific failing test failed: {e}")
        print(f"Error type: {type(e)}")
        return False
    
    return True

def test_import_order():
    """Test different import orders to see if that causes the issue."""
    print("Testing different import orders...")
    
    # Test 1: Import matplotlib first
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.cluster.hierarchy
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Import order 1: OK")
    except Exception as e:
        print(f"❌ Import order 1 failed: {e}")
        return False
    
    # Test 2: Import numpy first
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.cluster.hierarchy
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Import order 2: OK")
    except Exception as e:
        print(f"❌ Import order 2 failed: {e}")
        return False
    
    # Test 3: Import scipy first
    try:
        import scipy.cluster.hierarchy
        import numpy as np
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        print("✅ Import order 3: OK")
    except Exception as e:
        print(f"❌ Import order 3 failed: {e}")
        return False
    
    return True

def test_matplotlib_backend_switching():
    """Test matplotlib backend switching."""
    print("Testing matplotlib backend switching...")
    
    try:
        # Get current backend
        current_backend = plt.get_backend()
        print(f"Current backend: {current_backend}")
        
        # Try to switch backends
        plt.switch_backend('Agg')
        print(f"Switched to: {plt.get_backend()}")
        
        # Create figure with new backend
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        plt.close(fig)
        
        # Switch back
        plt.switch_backend(current_backend)
        print(f"Switched back to: {plt.get_backend()}")
        
        print("✅ Backend switching: OK")
        return True
    except Exception as e:
        print(f"❌ Backend switching failed: {e}")
        return False

def main():
    """Run all tests to identify the issue."""
    print("=" * 60)
    print("Test Environment Reproduction")
    print("=" * 60)
    
    tests = [
        simulate_test_environment,
        test_import_order,
        test_matplotlib_backend_switching
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
        print("The issue might be more subtle and related to:")
        print("- Test execution order")
        print("- Coverage collection")
        print("- Specific test combinations")

if __name__ == "__main__":
    main()
