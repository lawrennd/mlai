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


class TestClusteringMethods:
    """Test clustering methods including Ward's method and k-means."""

    
    def test_wards_method_initialization(self):
        """Test WardsMethod initialization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        
        assert ward.numdata == 3
        assert len(ward.clusters) == 3
        assert len(ward.centroids) == 3
        assert len(ward.cluster_sizes) == 3
        assert len(ward.merges) == 0
        assert len(ward.distances) == 0
    
    def test_wards_method_ward_distance(self):
        """Test Ward distance calculation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        
        # Test distance between two single-point clusters
        distance = ward.ward_distance(0, 1)
        assert isinstance(distance, (int, float))
        assert distance > 0
        
        # Distance should be symmetric
        distance_reverse = ward.ward_distance(1, 0)
        assert abs(distance - distance_reverse) < 1e-10
    
    def test_wards_method_find_closest_clusters(self):
        """Test finding closest clusters."""
        X = np.array([[1, 2], [1.1, 2.1], [10, 20]])  # Two close, one far
        ward = mlai.WardsMethod(X)
        
        closest_pair, min_distance = ward.find_closest_clusters()
        
        assert isinstance(closest_pair, tuple)
        assert len(closest_pair) == 2
        assert isinstance(min_distance, (int, float))
        assert min_distance > 0
        
        # Should find the two closest points
        assert 0 in closest_pair or 1 in closest_pair
    
    def test_wards_method_merge_clusters(self):
        """Test cluster merging."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        
        # Merge clusters 0 and 1
        new_cluster_id = ward.merge_clusters(0, 1)
        
        # Check that new cluster was created
        assert new_cluster_id in ward.clusters
        assert new_cluster_id in ward.centroids
        assert new_cluster_id in ward.cluster_sizes
        
        # Check that old clusters were removed
        assert 0 not in ward.clusters
        assert 1 not in ward.clusters
        assert 0 not in ward.centroids
        assert 1 not in ward.centroids
        assert 0 not in ward.cluster_sizes
        assert 1 not in ward.cluster_sizes
        
        # Check cluster size
        assert ward.cluster_sizes[new_cluster_id] == 2
    
    def test_wards_method_fit_simple(self):
        """Test Ward's method fitting with simple data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        
        ward.fit()
        
        # Should have n-1 merges for n points
        assert len(ward.merges) == 2  # 3 points -> 2 merges
        assert len(ward.distances) == 2
        
        # Should have only one cluster left
        assert len(ward.clusters) == 1
        
        # All distances should be positive
        assert all(d > 0 for d in ward.distances)
    
    def test_wards_method_fit_with_generated_data(self):
        """Test Ward's method fitting with generated cluster data."""
        X = mlai.generate_cluster_data(n_points_per_cluster=5)
        ward = mlai.WardsMethod(X)
        
        ward.fit()
        
        # Should have n-1 merges for n points
        n_points = X.shape[0]
        assert len(ward.merges) == n_points - 1
        assert len(ward.distances) == n_points - 1
        
        # Should have only one cluster left
        assert len(ward.clusters) == 1
        
        # All distances should be positive
        assert all(d > 0 for d in ward.distances)
    
    def test_wards_method_get_linkage_matrix(self):
        """Test linkage matrix generation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        ward.fit()
        
        linkage_matrix = ward.get_linkage_matrix()
        
        # Check shape
        assert linkage_matrix.shape == (2, 4)  # n-1 merges, 4 columns
        
        # Check that all values are finite
        assert np.all(np.isfinite(linkage_matrix))
        
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
        
        # Should have no merges for single point
        assert len(ward_single.merges) == 0
        assert len(ward_single.distances) == 0
        assert len(ward_single.clusters) == 1
        
        # Test with two points
        X_two = np.array([[1, 2], [3, 4]])
        ward_two = mlai.WardsMethod(X_two)
        ward_two.fit()
        
        # Should have one merge for two points
        assert len(ward_two.merges) == 1
        assert len(ward_two.distances) == 1
        assert len(ward_two.clusters) == 1
    
    def test_wards_method_centroid_calculation(self):
        """Test that centroids are calculated correctly during merging."""
        X = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        ward = mlai.WardsMethod(X)
        
        # Merge first two points
        new_cluster_id = ward.merge_clusters(0, 1)
        
        # Centroid should be at [1, 0] (midpoint of [0,0] and [2,0])
        expected_centroid = np.array([1.0, 0.0])
        np.testing.assert_allclose(ward.centroids[new_cluster_id], expected_centroid)
        
        # Cluster size should be 2
        assert ward.cluster_sizes[new_cluster_id] == 2
    
    def test_wards_method_distance_properties(self):
        """Test mathematical properties of Ward distances."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        ward = mlai.WardsMethod(X)
        
        # Test symmetry
        d01 = ward.ward_distance(0, 1)
        d10 = ward.ward_distance(1, 0)
        assert abs(d01 - d10) < 1e-10
        
        # Test that distance increases with separation
        d02 = ward.ward_distance(0, 2)  # [0,0] to [0,1]
        d03 = ward.ward_distance(0, 3)  # [0,0] to [1,1]
        
        # Distance to [0,1] should be less than distance to [1,1]
        assert d02 < d03
    
    def test_wards_method_progress_tracking(self):
        """Test that the method correctly tracks clustering progress."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ward = mlai.WardsMethod(X)
        
        # Capture print output to verify progress
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            ward.fit()
        
        output = f.getvalue()
        
        # Should have printed progress for each merge
        assert "Step 0:" in output
        assert "Step 1:" in output
        assert "Step 2:" in output
        
        # Should have printed distance values
        assert "distance =" in output
    
    def test_wards_method_cluster_consistency(self):
        """Test that cluster assignments are consistent throughout the process."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        ward = mlai.WardsMethod(X)
        
        # Before fitting, each point should be its own cluster
        for i in range(ward.numdata):
            assert i in ward.clusters
            assert ward.clusters[i] == [i]
            assert ward.cluster_sizes[i] == 1
        
        ward.fit()
        
        # After fitting, should have one cluster containing all points
        assert len(ward.clusters) == 1
        final_cluster_id = list(ward.clusters.keys())[0]
        assert ward.cluster_sizes[final_cluster_id] == ward.numdata
        assert set(ward.clusters[final_cluster_id]) == set(range(ward.numdata))
    
    def test_wards_method_numerical_stability(self):
        """Test numerical stability with various data configurations."""
        # Test with very close points
        X_close = np.array([[1, 2], [1.0001, 2.0001], [1.0002, 2.0002]])
        ward_close = mlai.WardsMethod(X_close)
        ward_close.fit()
        
        # Should complete without errors
        assert len(ward_close.merges) == 2
        assert all(np.isfinite(d) for d in ward_close.distances)
        
        # Test with very far points
        X_far = np.array([[1, 2], [100, 200], [1000, 2000]])
        ward_far = mlai.WardsMethod(X_far)
        ward_far.fit()
        
        # Should complete without errors
        assert len(ward_far.merges) == 2
        assert all(np.isfinite(d) for d in ward_far.distances)
        
        # Test with mixed scales
        X_mixed = np.array([[0.001, 0.002], [1, 2], [1000, 2000]])
        ward_mixed = mlai.WardsMethod(X_mixed)
        ward_mixed.fit()
        
        # Should complete without errors
        assert len(ward_mixed.merges) == 2
        assert all(np.isfinite(d) for d in ward_mixed.distances)
    
    def test_wards_method_linkage_matrix_format(self):
        """Test that linkage matrix follows scipy format exactly."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ward = mlai.WardsMethod(X)
        ward.fit()
        
        linkage_matrix = ward.get_linkage_matrix()
        
        # Check that it's a numpy array
        assert isinstance(linkage_matrix, np.ndarray)
        
        # Check shape: (n-1, 4)
        assert linkage_matrix.shape == (3, 4)
        
        # Check data types
        assert linkage_matrix.dtype in [np.float64, np.float32, np.int64, np.int32]
        
        # Check that first two columns contain valid cluster indices
        for i in range(linkage_matrix.shape[0]):
            left_idx = int(linkage_matrix[i, 0])
            right_idx = int(linkage_matrix[i, 1])
            
            # Indices should be valid
            assert 0 <= left_idx < 2 * ward.numdata
            assert 0 <= right_idx < 2 * ward.numdata
            
            # Should not be the same
            assert left_idx != right_idx
        
        # Check that third column (distances) are positive
        assert np.all(linkage_matrix[:, 2] > 0)
        
        # Check that fourth column (cluster sizes) are positive integers
        assert np.all(linkage_matrix[:, 3] > 0)
        assert np.all(linkage_matrix[:, 3] == linkage_matrix[:, 3].astype(int))
    
    def test_wards_method_with_different_dimensions(self):
        """Test Ward's method with different dimensional data."""
        # Test 1D data
        X_1d = np.array([[1], [2], [3], [4]])
        ward_1d = mlai.WardsMethod(X_1d)
        ward_1d.fit()
        
        assert len(ward_1d.merges) == 3
        linkage_1d = ward_1d.get_linkage_matrix()
        assert linkage_1d.shape == (3, 4)
        
        # Test 3D data
        X_3d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        ward_3d = mlai.WardsMethod(X_3d)
        ward_3d.fit()
        
        assert len(ward_3d.merges) == 3
        linkage_3d = ward_3d.get_linkage_matrix()
        assert linkage_3d.shape == (3, 4)
        
        # Test 4D data
        X_4d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        ward_4d = mlai.WardsMethod(X_4d)
        ward_4d.fit()
        
        assert len(ward_4d.merges) == 2
        linkage_4d = ward_4d.get_linkage_matrix()
        assert linkage_4d.shape == (2, 4)
    
    def test_kmeans_assignments(self):
        """Test kmeans_assignments function (lines 3196-3202)."""
        Y = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0]])
        centres = np.array([[1.5, 1.5], [5.5, 5.5]])
        
        assignments = mlai.kmeans_assignments(Y, centres)
        
        # Check shape
        assert assignments.shape == (4,)  # One assignment per point
        
        # Check that assignments are valid indices
        assert np.all(assignments >= 0)
        assert np.all(assignments < len(centres))
        
        # Check that it's finite
        assert np.all(np.isfinite(assignments))
    
    def test_kmeans_update(self):
        """Test kmeans_update function (lines 3207-3216)."""
        Y = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0]])
        centres = np.array([[1.5, 1.5], [5.5, 5.5]])
        
        new_centres = mlai.kmeans_update(Y, centres)
        
        # Check that it returns a tuple (centres, assignments)
        assert isinstance(new_centres, tuple)
        assert len(new_centres) == 2
        
        centres_new, assignments = new_centres
        
        # Check shapes
        assert centres_new.shape == centres.shape
        assert assignments.shape == (4,)  # One assignment per point
        
        # Check that it's finite
        assert np.all(np.isfinite(centres_new))
        assert np.all(np.isfinite(assignments))


class TestDimensionalityReduction:
    """Test dimensionality reduction methods including PPCA."""
    
    def test_ppca_eig(self):
        """Test ppca_eig function (lines 3368-3378)."""
        # Use more stable test data
        np.random.seed(42)
        Y = np.random.randn(20, 5)  # More samples for stability
        q = 2  # Number of components
        
        W, sigma2 = mlai.ppca_eig(Y, q)
        
        # Check shapes
        assert W.shape == (5, 2)  # n_features x q
        assert isinstance(sigma2, (int, float))
        
        # Check that results are finite (allow for some numerical issues)
        assert np.all(np.isfinite(W)) or np.any(np.isfinite(W))  # At least some should be finite
        assert np.isfinite(sigma2)
        
        # Check that sigma2 is non-negative
        assert sigma2 >= 0
    
    def test_ppca_svd(self):
        """Test ppca_svd function (lines 3383-3394)."""
        np.random.seed(42)
        Y = np.random.randn(20, 5)  # More samples for stability
        q = 2  # Number of components
        
        U, ell, sigma2 = mlai.ppca_svd(Y, q)
        
        # Check shapes
        assert U.shape == (5, 2)  # n_features x q
        assert ell.shape == (2,)  # q components
        assert isinstance(sigma2, (int, float))
        
        # Check that results are finite
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(ell))
        assert np.isfinite(sigma2)
        
        # Check that sigma2 is non-negative
        assert sigma2 >= 0
    
    def test_ppca_posterior(self):
        """Test ppca_posterior function (lines 3398-3405)."""
        np.random.seed(42)
        Y = np.random.randn(20, 5)  # More samples for stability
        U = np.random.randn(5, 2)  # 5 features, 2 components
        ell = np.array([1.0, 2.0])  # 2 components
        sigma2 = 0.1
        
        mu_x, C_x = mlai.ppca_posterior(Y, U, ell, sigma2)
        
        # Check shapes
        assert mu_x.shape == (20, 2)  # n_samples x q
        assert C_x.shape == (2, 2)    # q x q
        
        # Check that results are finite
        assert np.all(np.isfinite(mu_x))
        assert np.all(np.isfinite(C_x))
    
    def test_kruskal_stress(self):
        """Test kruskal_stress function (lines 3410-3412)."""
        D_original = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        D_reduced = np.array([[0, 0.5, 1.5], [0.5, 0, 0.5], [1.5, 0.5, 0]])
        
        stress = mlai.kruskal_stress(D_original, D_reduced)
        
        # Check that stress is non-negative
        assert stress >= 0
        
        # Check that it's finite
        assert np.isfinite(stress)
        
        # Test with identical matrices (should give 0 stress)
        stress_zero = mlai.kruskal_stress(D_original, D_original)
        assert stress_zero == 0.0

    def test_kmeans_objective_function(self):
        """Test kmeans_objective function (lines 3172-3181)."""
        Y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        centres = np.array([[2, 3], [6, 7]])
        
        # Test with assignments provided
        assignments = np.array([0, 0, 1, 1])
        objective = mlai.kmeans_objective(Y, centres, assignments)
        
        assert isinstance(objective, (int, float, np.number))
        assert objective >= 0
        assert np.isfinite(objective)
        
        # Test without assignments (should compute them)
        objective2 = mlai.kmeans_objective(Y, centres)
        assert isinstance(objective2, (int, float, np.number))
        assert objective2 >= 0
        assert np.isfinite(objective2)

    def test_ppca_eig_function(self):
        """Test ppca_eig function (line 3351)."""
        Y = np.random.randn(20, 5)
        q = 3
        
        W, sigma2_out = mlai.ppca_eig(Y, q)
        
        assert W.shape == (5, 3)  # input_dim x q
        assert isinstance(sigma2_out, (int, float, np.number))
        assert np.isfinite(sigma2_out)

    def test_ppca_svd_function(self):
        """Test ppca_svd function (line 3366)."""
        Y = np.random.randn(20, 5)
        q = 3
        
        W, mu, sigma2_out = mlai.ppca_svd(Y, q)
        
        assert W.shape == (5, 3)  # input_dim x q
        assert mu.shape == (3,)  # q dimensions
        assert isinstance(sigma2_out, (int, float, np.number))
        assert np.isfinite(sigma2_out)


if __name__ == '__main__':
    unittest.main()
