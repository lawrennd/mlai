"""
mlai.data
=========

Data generation and manipulation utilities.

This module contains functions for generating synthetic datasets, loading data,
and performing data preprocessing tasks for machine learning applications.

Features:
- Synthetic data generation (clusters, swiss roll, etc.)
- Data loading utilities (PGM files, etc.)
- Data preprocessing and transformation
- Dataset creation for educational purposes

Note: This module is part of the refactoring process to organize data-related
functionality from the main mlai.py file.
"""

import numpy as np

__all__ = [
    # Data generation functions
    'generate_cluster_data',
    'generate_swiss_roll',
]

def generate_cluster_data(n_points_per_cluster=30):
    """Generate synthetic data with clear cluster structure for educational purposes"""
    # Define cluster centres in 2D space
    cluster_centres = np.array([[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]])
    
    # Generate data points around each center
    data_points = []
    for center in cluster_centres:
        # Generate points with some spread around each center
        cluster_points = np.random.normal(loc=center, scale=0.8, size=(n_points_per_cluster, 2))
        data_points.append(cluster_points)
    
    return np.vstack(data_points)

def generate_swiss_roll(n_points=1000, noise=0.05):
    """Generate Swiss roll dataset"""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
    y = 21 * np.random.rand(n_points)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X.T, t
