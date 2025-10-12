#!/usr/bin/env python3
"""
Tests for dimensionality reduction and clustering functions in mlai.

This module tests the clustering and dimensionality reduction functions that will be 
moved to dimred.py in the refactoring process.
"""

import unittest
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch, MagicMock

# Import mlai modules
import mlai.mlai as mlai


class TestWardsMethod:
    """Test Ward's clustering method."""
    
    def test_wards_method_basic_functionality(self):
        """Test basic Ward's method functionality."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ward = mlai.WardsMethod(X)
        ward.fit()
        
        # Check that we get a linkage matrix
        linkage_matrix = ward.get_linkage_matrix()
        assert linkage_matrix.shape[1] == 4  # Should have 4 columns
        
        # Check that distances are positive
        assert np.all(linkage_matrix[:, 2] > 0)
        
        # Check that cluster sizes are positive
        assert np.all(linkage_matrix[:, 3] > 0)
    
    def test_wards_method_linkage_matrix_compatibility(self):
        """Test that linkage matrix is compatible with scipy."""
        from scipy.cluster.hierarchy import dendrogram
        import matplotlib.pyplot as plt

        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        ward.fit()

        linkage_matrix = ward.get_linkage_matrix()

        # Should be able to create dendrogram without errors
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            dendrogram(linkage_matrix, ax=ax)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Linkage matrix not compatible with scipy: {e}")
        finally:
            plt.close('all')
    
    def test_wards_method_compare_with_scipy(self):
        """Test comparison with scipy's Ward linkage."""
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        import matplotlib.pyplot as plt
        
        try:
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            ward = mlai.WardsMethod(X)
            ward.fit()
            
            our_linkage = ward.get_linkage_matrix()
            scipy_linkage_result = scipy_linkage(X, method='ward')
            
            # Both should have same shape
            assert our_linkage.shape == scipy_linkage_result.shape
            
            # Both should have positive distances
            assert np.all(our_linkage[:, 2] > 0)
            assert np.all(scipy_linkage_result[:, 2] > 0)
            
            # Both should have positive cluster sizes
            assert np.all(our_linkage[:, 3] > 0)
            assert np.all(scipy_linkage_result[:, 3] > 0)
        finally:
            plt.close('all')
    
    def test_wards_method_edge_cases(self):
        """Test edge cases for Ward's method."""
        # Test with single point
        X_single = np.array([[1, 2]])
        ward_single = mlai.WardsMethod(X_single)
        ward_single.fit()
        
        # Should handle single point gracefully
        linkage_single = ward_single.get_linkage_matrix()
        assert linkage_single.shape[0] == 0  # No merges for single point
        
        # Test with two points
        X_two = np.array([[1, 2], [3, 4]])
        ward_two = mlai.WardsMethod(X_two)
        ward_two.fit()
        
        linkage_two = ward_two.get_linkage_matrix()
        assert linkage_two.shape[0] == 1  # One merge for two points
        assert linkage_two[0, 2] > 0  # Positive distance


if __name__ == '__main__':
    unittest.main()
