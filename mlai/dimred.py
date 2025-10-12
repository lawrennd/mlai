"""
Dimensionality Reduction Module

This module contains dimensionality reduction and clustering implementations including:
- Clustering algorithms (kmeans_assignments, kmeans_update, kmeans_objective, WardsMethod)
- Dimensionality reduction (ppca_eig, ppca_svd, ppca_posterior, kruskal_stress)

TODO: Extract from mlai.py during refactoring
"""

import numpy as np
import scipy.linalg as la
from .utils import dist2

__all__ = [
    # Clustering Algorithms
    'kmeans_assignments',
    'kmeans_update',
    'kmeans_objective',
    'WardsMethod',
    
    # Dimensionality Reduction
    'ppca_eig',
    'ppca_svd', 
    'ppca_posterior',
    'kruskal_stress',
]





class ClusterModel():
    pass

        
def kmeans_assignments(Y, centres):
    """Assign each point to nearest centre"""
    sq_distances = ((Y[:, np.newaxis] - centres[np.newaxis, :])**2).sum(axis=2)
    return np.argmin(sq_distances, axis=1)
    
def kmeans_update(Y, centres):
    """Perform an update of centre locations for k-means algorithm"""
    assignments = kmeans_assignments(Y, centres)
    
    # Update centres to be mean of assigned points
    new_centres = np.array([Y[assignments == k].mean(axis=0) 
                           for k in range(len(centres))])
    
    return new_centres, assignments

def kmeans_objective(Y, centres, assignments=None):
    """Calculate the k-means objective function (sum of squared distances)"""

    if assignments is None:
        assignments = kmeans_assignments(Y, centres)
        
    total_error = 0
    for k in range(len(centres)):
        cluster_points = Y[assignments == k]
        if len(cluster_points) > 0:
            sq_distances = dist2(cluster_points, centres[k:k+1, :])
            total_error += np.sum(sq_distances)
    return total_error


class WardsMethod(ClusterModel):
    def __init__(self, X):
        """
        Simple implementation of Ward's hierarchical clustering
        
        Parameters:
        X: numpy array of shape (n_samples, n_features)
        """
        self.numdata = len(X)
        
        # Initialize each point as its own cluster
        self.clusters = {i: [i] for i in range(self.numdata)}
        self.centroids = {i: X[i].copy() for i in range(self.numdata)}
        self.cluster_sizes = {i: 1 for i in range(self.numdata)}
        
        # Track merges for dendrogram
        self.merges = []
        self.distances = []
        
    def ward_distance(self, cluster_a, cluster_b):
        """
        Calculate Ward distance between two clusters
        """
        n_a = self.cluster_sizes[cluster_a]
        n_b = self.cluster_sizes[cluster_b]
        
        centroid_a = self.centroids[cluster_a]
        centroid_b = self.centroids[cluster_b]
        
        # Ward distance formula
        weight = (n_a * n_b) / (n_a + n_b)
        distance = np.sum((centroid_a - centroid_b) ** 2)
        
        return np.sqrt(weight * distance)
    
    def find_closest_clusters(self):
        """
        Find the pair of clusters with minimum Ward distance
        """
        min_distance = float('inf')
        closest_pair = None
        
        cluster_ids = list(self.clusters.keys())
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster_a = cluster_ids[i]
                cluster_b = cluster_ids[j]
                
                distance = self.ward_distance(cluster_a, cluster_b)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (cluster_a, cluster_b)
        
        return closest_pair, min_distance
    
    def merge_clusters(self, cluster_a, cluster_b):
        """
        Merge two clusters and update centroids
        """
        n_a = self.cluster_sizes[cluster_a]
        n_b = self.cluster_sizes[cluster_b]
        
        # Calculate new centroid as weighted mean
        centroid_a = self.centroids[cluster_a]
        centroid_b = self.centroids[cluster_b]
        new_centroid = (n_a * centroid_a + n_b * centroid_b) / (n_a + n_b)
        
        # Create new cluster ID
        new_cluster_id = max(self.clusters.keys()) + 1
        
        # Merge point lists
        new_cluster_points = self.clusters[cluster_a] + self.clusters[cluster_b]
        
        # Update data structures
        self.clusters[new_cluster_id] = new_cluster_points
        self.centroids[new_cluster_id] = new_centroid
        self.cluster_sizes[new_cluster_id] = n_a + n_b
        
        # Remove old clusters
        del self.clusters[cluster_a]
        del self.clusters[cluster_b]
        del self.centroids[cluster_a]
        del self.centroids[cluster_b]
        del self.cluster_sizes[cluster_a]
        del self.cluster_sizes[cluster_b]
        
        return new_cluster_id
    
    def fit(self):
        """
        Perform Ward's hierarchical clustering
        """
        step = 0
        
        while len(self.clusters) > 1:
            # Find closest pair
            (cluster_a, cluster_b), distance = self.find_closest_clusters()
            
            print(f"Step {step}: Merging clusters {cluster_a} and {cluster_b}, "
                  f"distance = {distance:.3f}")
            
            # Record merge for dendrogram
            self.merges.append([cluster_a, cluster_b])
            self.distances.append(distance)
            
            # Merge clusters
            new_cluster_id = self.merge_clusters(cluster_a, cluster_b)
            
            step += 1
        
        return self
    
    def get_linkage_matrix(self):
        """
        Convert to scipy linkage matrix format for dendrogram plotting
        """
        linkage_matrix = []
        
        # Track cluster sizes and scipy IDs
        cluster_sizes = {i: 1 for i in range(self.numdata)}
        cluster_to_scipy = {i: i for i in range(self.numdata)}
        next_scipy_id = self.numdata
        
        for i, (merge, distance) in enumerate(zip(self.merges, self.distances)):
            cluster_a, cluster_b = merge
            
            # Get scipy IDs for the clusters being merged
            scipy_a = cluster_to_scipy.get(cluster_a, cluster_a)
            scipy_b = cluster_to_scipy.get(cluster_b, cluster_b)
            
            # Get cluster sizes
            size_a = cluster_sizes.get(cluster_a, 1)
            size_b = cluster_sizes.get(cluster_b, 1)
            
            # Create the linkage matrix row
            linkage_matrix.append([scipy_a, scipy_b, distance, size_a + size_b])
            
            # Update tracking for the new merged cluster
            new_cluster_id = self.numdata + i
            cluster_sizes[new_cluster_id] = size_a + size_b
            cluster_to_scipy[new_cluster_id] = new_cluster_id
        
        return np.array(linkage_matrix)

def ppca_eig(Y, q):
    """Perform probabilistic principle component analysis"""
    
    Y_cent = Y - Y.mean(0)

    # Comute covariance
    S = np.dot(Y_cent.T, Y_cent)/Y.shape[0]
    lambd, U = np.linalg.eig(S)

    # Choose number of eigenvectors
    sigma2 = np.sum(lambd[q:])/(Y.shape[1]-q)
    l = np.sqrt(lambd[:q]-sigma2)
    W = U[:, :q]*l[None, :]
    return W, sigma2

def ppca_svd(Y, q, center=True):
    """Probabilistic PCA through singular value decomposition"""
    # remove mean
    if center:
        Y_cent = Y - Y.mean(0)
    else:
        Y_cent = Y
    import scipy as sp
    # Comute singluar values, discard 'R' as we will assume orthogonal
    U, sqlambd, _ = sp.linalg.svd(Y_cent.T,full_matrices=False)
    lambd = (sqlambd**2)/Y.shape[0]
    # Compute residual and extract eigenvectors
    sigma2 = np.sum(lambd[q:])/(Y.shape[1]-q)
    ell = np.sqrt(lambd[:q]-sigma2)
    return U[:, :q], ell, sigma2

def ppca_posterior(Y, U, ell, sigma2, center=True):
    """Posterior computation for the latent variables given the eigendecomposition."""
    if center:
        Y_cent = Y - Y.mean(0)
    else:
        Y_cent = Y
    C_x = np.diag(sigma2/(sigma2+ell**2))
    d = ell/(sigma2+ell**2)
    mu_x = np.dot(Y_cent, U)*d[None, :]
    return mu_x, C_x

    
def kruskal_stress(D_original, D_reduced):
    """Compute Kruskal's stress between original and reduced distances"""
    numerator = np.sum((D_original - D_reduced)**2)
    denominator = np.sum(D_original**2)
    return np.sqrt(numerator/denominator)


