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
    'generate_sequence_data',
    'generate_arithmetic_sequences',
    'generate_pattern_sequences',
    'generate_text_sequences',
    'create_image_data',
    'create_synthetic_data',
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


def generate_sequence_data(n_samples=200, seq_length=10, vocab_size=30, pattern_type='next_token'):
    """
    Generate interesting sequence data for transformer training.
    
    Parameters
    ----------
    n_samples : int
        Number of sequences to generate
    seq_length : int
        Length of each sequence
    vocab_size : int
        Size of vocabulary (number of unique tokens)
    pattern_type : str
        Type of pattern to generate ('next_token', 'arithmetic', 'pattern', 'text')
        
    Returns
    -------
    X : np.ndarray
        Input sequences of shape (n_samples, seq_length)
    y : np.ndarray
        Target sequences of shape (n_samples, seq_length)
    """
    np.random.seed(24)  # For reproducibility
    
    if pattern_type == 'next_token':
        return _generate_next_token_sequences(n_samples, seq_length, vocab_size)
    elif pattern_type == 'arithmetic':
        return generate_arithmetic_sequences(n_samples, seq_length, vocab_size)
    elif pattern_type == 'pattern':
        return generate_pattern_sequences(n_samples, seq_length, vocab_size)
    elif pattern_type == 'text':
        return generate_text_sequences(n_samples, seq_length, vocab_size)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")


def _generate_next_token_sequences(n_samples, seq_length, vocab_size):
    """Generate sequences where target is next token (standard language modeling)."""
    X = np.random.randint(0, vocab_size, (n_samples, seq_length))
    y = np.roll(X, -1, axis=1)
    y[:, -1] = 0  # Padding token for last position
    return X, y


def generate_arithmetic_sequences(n_samples=200, seq_length=8, vocab_size=20):
    """
    Generate arithmetic sequence data (e.g., 2, 4, 6, 8, ...).
    
    This creates sequences where each number follows an arithmetic pattern,
    making it interesting for transformers to learn mathematical relationships.
    """
    np.random.seed(24)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random starting value and step
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        
        # Generate arithmetic sequence
        sequence = [start + i * step for i in range(seq_length + 1)]
        
        # Clip to vocabulary size
        sequence = [min(x, vocab_size - 1) for x in sequence]
        
        X.append(sequence[:-1])
        y.append(sequence[1:])
    
    return np.array(X), np.array(y)


def generate_pattern_sequences(n_samples=200, seq_length=10, vocab_size=20):
    """
    Generate sequences with repeating patterns (e.g., A, B, C, A, B, C, ...).
    
    This tests the transformer's ability to learn and maintain patterns
    across the sequence.
    """
    np.random.seed(24)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random pattern length (2-4)
        pattern_length = np.random.randint(2, 5)
        
        # Generate random pattern
        pattern = np.random.randint(0, vocab_size, pattern_length)
        
        # Repeat pattern to fill sequence
        full_sequence = []
        for i in range(seq_length + 1):
            full_sequence.append(pattern[i % pattern_length])
        
        X.append(full_sequence[:-1])
        y.append(full_sequence[1:])
    
    return np.array(X), np.array(y)


def generate_text_sequences(n_samples=200, seq_length=12, vocab_size=26):
    """
    Generate text-like sequences with word boundaries and structure.
    
    This simulates natural language with spaces, punctuation, and
    word-like structures for more realistic transformer training.
    """
    np.random.seed(24)
    
    # Define special tokens
    SPACE = 0
    PERIOD = 1
    COMMA = 2
    LETTERS_START = 3
    
    X = []
    y = []
    
    for _ in range(n_samples):
        sequence = []
        word_length = 0
        max_word_length = 6
        
        for i in range(seq_length + 1):
            if word_length == 0:
                # Start of new word
                if np.random.random() < 0.1 and i > 0:
                    # Add punctuation
                    sequence.append(np.random.choice([PERIOD, COMMA]))
                else:
                    # Add letter
                    sequence.append(LETTERS_START + np.random.randint(0, vocab_size - LETTERS_START))
                word_length = 1
            elif word_length >= max_word_length or np.random.random() < 0.3:
                # End word with space
                sequence.append(SPACE)
                word_length = 0
            else:
                # Continue word
                sequence.append(LETTERS_START + np.random.randint(0, vocab_size - LETTERS_START))
                word_length += 1
        
        X.append(sequence[:-1])
        y.append(sequence[1:])
    
    return np.array(X), np.array(y)


def create_image_data(n_samples=100, image_size=28, n_classes=3):
    """Create synthetic image data for CNN demonstration.
    
    This function generates synthetic images with different geometric patterns
    that can be used to test and demonstrate CNN architectures.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of images to generate, by default 100
    image_size : int, optional
        Size of each image (image_size x image_size), by default 28
    n_classes : int, optional
        Number of different pattern classes, by default 3
        
    Returns
    -------
    X : np.ndarray
        Array of shape (n_samples, 1, image_size, image_size) containing the images
    y : np.ndarray
        Array of shape (n_samples,) containing the class labels
        
    Notes
    -----
    The function creates three types of synthetic patterns:
    - Class 0: Horizontal lines
    - Class 1: Vertical lines  
    - Class 2: Diagonal patterns
    
    Each image is generated with some noise to make the learning task more realistic.
    """
    np.random.seed(24)
    
    # Create different types of synthetic images
    X = []
    y = []
    
    for i in range(n_samples):
        # Create a synthetic image with different patterns
        image = np.zeros((image_size, image_size))
        
        # Add some geometric patterns
        if i % 3 == 0:
            # Horizontal lines
            for row in range(5, image_size-5, 3):
                image[row:row+2, 5:image_size-5] = 1.0
        elif i % 3 == 1:
            # Vertical lines  
            for col in range(5, image_size-5, 3):
                image[5:image_size-5, col:col+2] = 1.0
        else:
            # Diagonal patterns
            for d in range(0, image_size, 4):
                for j in range(max(0, d-image_size+1), min(d+1, image_size)):
                    if j < image_size and d-j < image_size:
                        image[j, d-j] = 1.0
        
        # Add some noise
        image += 0.1 * np.random.randn(image_size, image_size)
        
        # Reshape to (channels, height, width) format
        image = image.reshape(1, image_size, image_size)
        X.append(image)
        
        # Create labels based on pattern type
        y.append(i % n_classes)
    
    return np.array(X), np.array(y)

def create_synthetic_data(n_samples=100, task='regression'):
    """Create synthetic datasets for demonstration."""
    np.random.seed(24)
    
    if task == 'regression':
        # Non-linear regression: y = x1^2 + x2^2 + noise
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1) + 0.1 * np.random.randn(n_samples, 1)
        return X, y
    
    elif task == 'classification':
        # Binary classification: x1^2 + x2^2 > 1
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0]**2 + X[:, 1]**2) > 1.0).astype(float).reshape(-1, 1)
        return X, y
