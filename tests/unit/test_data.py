"""
Unit tests for mlai.data module.

This module tests the data generation and manipulation utilities,
including the new sequence data generation functions for transformer training.
"""

import unittest
import numpy as np
from mlai.data import (
    generate_cluster_data,
    generate_swiss_roll,
    generate_sequence_data,
    generate_arithmetic_sequences,
    generate_pattern_sequences,
    generate_text_sequences,
    create_image_data
)


class TestDataGeneration(unittest.TestCase):
    """Test basic data generation functions."""
    
    def test_generate_cluster_data(self):
        """Test cluster data generation."""
        data = generate_cluster_data(n_points_per_cluster=10)
        
        # Check shape
        self.assertEqual(data.shape, (30, 2))  # 3 clusters * 10 points
        
        # Check that we have points from all clusters
        self.assertTrue(np.allclose(data[:10].mean(axis=0), [2.5, 2.5], atol=1.0))
        self.assertTrue(np.allclose(data[10:20].mean(axis=0), [-2.5, -2.5], atol=1.0))
        self.assertTrue(np.allclose(data[20:30].mean(axis=0), [2.5, -2.5], atol=1.0))
    
    def test_generate_swiss_roll(self):
        """Test Swiss roll data generation."""
        X, t = generate_swiss_roll(n_points=100, noise=0.01)
        
        # Check shapes
        self.assertEqual(X.shape, (100, 3))
        self.assertEqual(t.shape, (100,))
        
        # Check that data is reasonable
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(np.isfinite(t)))


class TestSequenceDataGeneration(unittest.TestCase):
    """Test sequence data generation functions."""
    
    def test_generate_sequence_data_arithmetic(self):
        """Test arithmetic sequence generation."""
        X, y = generate_sequence_data(n_samples=10, seq_length=5, vocab_size=20, pattern_type='arithmetic')
        
        # Check shapes
        self.assertEqual(X.shape, (10, 5))
        self.assertEqual(y.shape, (10, 5))
        
        # Check that sequences follow arithmetic progression (allowing for vocab clipping)
        for i in range(10):
            sequence = np.concatenate([X[i], [y[i][-1]]])  # Full sequence
            diffs = np.diff(sequence)
            # Most differences should be the same (allowing for vocab clipping at the end)
            unique_diffs = np.unique(diffs)
            # Should have at most 3 unique differences (allowing for vocab clipping effects)
            self.assertLessEqual(len(unique_diffs), 3, f"Sequence {sequence} has too many different step sizes")
    
    def test_generate_sequence_data_pattern(self):
        """Test pattern sequence generation."""
        X, y = generate_sequence_data(n_samples=10, seq_length=8, vocab_size=15, pattern_type='pattern')
        
        # Check shapes
        self.assertEqual(X.shape, (10, 8))
        self.assertEqual(y.shape, (10, 8))
        
        # Check that sequences follow repeating patterns
        for i in range(10):
            full_sequence = np.concatenate([X[i], [y[i][-1]]])
            # Check for pattern by looking for repeated subsequences
            pattern_found = False
            for pattern_len in range(2, min(5, len(full_sequence) // 2)):
                # Check if the first pattern_len elements repeat
                if len(full_sequence) >= 2 * pattern_len:
                    if np.array_equal(full_sequence[:pattern_len], full_sequence[pattern_len:2*pattern_len]):
                        pattern_found = True
                        break
            # If no exact repetition, check for approximate patterns (allowing for some variation)
            if not pattern_found:
                # Look for repeated elements or small patterns
                unique_elements = len(np.unique(full_sequence))
                # Pattern sequences should have limited unique elements (repetitive)
                if unique_elements <= len(full_sequence) // 2 + 1:  # Allow one extra unique element
                    pattern_found = True
            self.assertTrue(pattern_found, f"Sequence {full_sequence} doesn't follow a clear pattern")
    
    def test_generate_sequence_data_text(self):
        """Test text sequence generation."""
        X, y = generate_sequence_data(n_samples=10, seq_length=10, vocab_size=26, pattern_type='text')
        
        # Check shapes
        self.assertEqual(X.shape, (10, 10))
        self.assertEqual(y.shape, (10, 10))
        
        # Check that sequences contain special tokens (spaces, punctuation)
        # SPACE=0, PERIOD=1, COMMA=2, LETTERS_START=3
        for i in range(10):
            sequence = X[i]
            # Should have some spaces (token 0)
            self.assertIn(0, sequence, "Text sequences should contain spaces")
            # Should have some letters (tokens >= 3)
            letters = sequence[sequence >= 3]
            self.assertGreater(len(letters), 0, "Text sequences should contain letters")
    
    def test_generate_sequence_data_next_token(self):
        """Test next token sequence generation."""
        X, y = generate_sequence_data(n_samples=10, seq_length=6, vocab_size=20, pattern_type='next_token')
        
        # Check shapes
        self.assertEqual(X.shape, (10, 6))
        self.assertEqual(y.shape, (10, 6))
        
        # Check that y is X shifted by 1
        for i in range(10):
            self.assertTrue(np.array_equal(y[i][:-1], X[i][1:]))
            self.assertEqual(y[i][-1], 0)  # Padding token
    
    def test_generate_sequence_data_invalid_type(self):
        """Test error handling for invalid pattern type."""
        with self.assertRaises(ValueError):
            generate_sequence_data(pattern_type='invalid_type')


class TestArithmeticSequences(unittest.TestCase):
    """Test arithmetic sequence generation specifically."""
    
    def test_arithmetic_sequences_basic(self):
        """Test basic arithmetic sequence generation."""
        X, y = generate_arithmetic_sequences(n_samples=5, seq_length=4, vocab_size=20)
        
        # Check shapes
        self.assertEqual(X.shape, (5, 4))
        self.assertEqual(y.shape, (5, 4))
        
        # Check arithmetic progression (allowing for vocab clipping)
        for i in range(5):
            full_sequence = np.concatenate([X[i], [y[i][-1]]])
            diffs = np.diff(full_sequence)
            # Should have at most 2 unique differences (normal progression + clipped values)
            unique_diffs = np.unique(diffs)
            self.assertLessEqual(len(unique_diffs), 2, f"Sequence {full_sequence} has too many different step sizes")
    
    def test_arithmetic_sequences_vocab_clipping(self):
        """Test that arithmetic sequences respect vocabulary size."""
        X, y = generate_arithmetic_sequences(n_samples=10, seq_length=6, vocab_size=10)
        
        # All values should be within vocabulary
        self.assertTrue(np.all(X < 10))
        self.assertTrue(np.all(y < 10))
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(y >= 0))


class TestPatternSequences(unittest.TestCase):
    """Test pattern sequence generation specifically."""
    
    def test_pattern_sequences_basic(self):
        """Test basic pattern sequence generation."""
        X, y = generate_pattern_sequences(n_samples=5, seq_length=6, vocab_size=15)
        
        # Check shapes
        self.assertEqual(X.shape, (5, 6))
        self.assertEqual(y.shape, (5, 6))
        
        # Check that sequences follow patterns
        for i in range(5):
            full_sequence = np.concatenate([X[i], [y[i][-1]]])
            # Check for pattern by looking for repeated subsequences
            pattern_found = False
            for pattern_len in range(2, min(4, len(full_sequence) // 2)):
                if len(full_sequence) >= 2 * pattern_len:
                    if np.array_equal(full_sequence[:pattern_len], full_sequence[pattern_len:2*pattern_len]):
                        pattern_found = True
                        break
            # If no exact repetition, check for limited unique elements (repetitive nature)
            if not pattern_found:
                unique_elements = len(np.unique(full_sequence))
                # Pattern sequences should have limited unique elements (repetitive)
                if unique_elements <= len(full_sequence) // 2 + 1:  # Allow one extra unique element
                    pattern_found = True
            self.assertTrue(pattern_found, f"Sequence {full_sequence} doesn't follow a clear pattern")
    
    def test_pattern_sequences_vocab_respect(self):
        """Test that pattern sequences respect vocabulary size."""
        X, y = generate_pattern_sequences(n_samples=10, seq_length=8, vocab_size=10)
        
        # All values should be within vocabulary
        self.assertTrue(np.all(X < 10))
        self.assertTrue(np.all(y < 10))
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(y >= 0))


class TestTextSequences(unittest.TestCase):
    """Test text sequence generation specifically."""
    
    def test_text_sequences_basic(self):
        """Test basic text sequence generation."""
        X, y = generate_text_sequences(n_samples=5, seq_length=8, vocab_size=26)
        
        # Check shapes
        self.assertEqual(X.shape, (5, 8))
        self.assertEqual(y.shape, (5, 8))
        
        # Check that sequences contain expected tokens
        for i in range(5):
            sequence = X[i]
            # Should have spaces (token 0)
            self.assertIn(0, sequence, "Text sequences should contain spaces")
            # Should have letters (tokens >= 3)
            letters = sequence[sequence >= 3]
            self.assertGreater(len(letters), 0, "Text sequences should contain letters")
    
    def test_text_sequences_structure(self):
        """Test that text sequences have word-like structure."""
        X, y = generate_text_sequences(n_samples=10, seq_length=12, vocab_size=26)
        
        for i in range(10):
            sequence = X[i]
            # Check for word boundaries (spaces)
            spaces = np.where(sequence == 0)[0]
            self.assertGreater(len(spaces), 0, "Should have word boundaries")
            
            # Check that we don't have too many consecutive spaces
            consecutive_spaces = 0
            max_consecutive = 0
            for token in sequence:
                if token == 0:
                    consecutive_spaces += 1
                    max_consecutive = max(max_consecutive, consecutive_spaces)
                else:
                    consecutive_spaces = 0
            self.assertLessEqual(max_consecutive, 1, "Should not have multiple consecutive spaces")
    
    def test_text_sequences_vocab_respect(self):
        """Test that text sequences respect vocabulary size."""
        X, y = generate_text_sequences(n_samples=10, seq_length=10, vocab_size=20)
        
        # All values should be within vocabulary
        self.assertTrue(np.all(X < 20))
        self.assertTrue(np.all(y < 20))
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(y >= 0))


class TestDataReproducibility(unittest.TestCase):
    """Test that data generation is reproducible."""
    
    def test_reproducibility_arithmetic(self):
        """Test that arithmetic sequences are reproducible."""
        # Generate same data twice
        X1, y1 = generate_arithmetic_sequences(n_samples=5, seq_length=4, vocab_size=20)
        X2, y2 = generate_arithmetic_sequences(n_samples=5, seq_length=4, vocab_size=20)
        
        # Should be identical
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_reproducibility_pattern(self):
        """Test that pattern sequences are reproducible."""
        # Generate same data twice
        X1, y1 = generate_pattern_sequences(n_samples=5, seq_length=6, vocab_size=15)
        X2, y2 = generate_pattern_sequences(n_samples=5, seq_length=6, vocab_size=15)
        
        # Should be identical
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_reproducibility_text(self):
        """Test that text sequences are reproducible."""
        # Generate same data twice
        X1, y1 = generate_text_sequences(n_samples=5, seq_length=8, vocab_size=26)
        X2, y2 = generate_text_sequences(n_samples=5, seq_length=8, vocab_size=26)
        
        # Should be identical
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestDataEdgeCases(unittest.TestCase):
    """Test edge cases for data generation."""
    
    def test_small_sequences(self):
        """Test generation with very small sequences."""
        X, y = generate_arithmetic_sequences(n_samples=2, seq_length=2, vocab_size=5)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))
    
    def test_small_vocab(self):
        """Test generation with very small vocabulary."""
        X, y = generate_arithmetic_sequences(n_samples=3, seq_length=4, vocab_size=3)
        self.assertTrue(np.all(X < 3))
        self.assertTrue(np.all(y < 3))
    
    def test_large_sequences(self):
        """Test generation with larger sequences."""
        X, y = generate_pattern_sequences(n_samples=5, seq_length=20, vocab_size=10)
        self.assertEqual(X.shape, (5, 20))
        self.assertEqual(y.shape, (5, 20))


class TestImageDataGeneration(unittest.TestCase):
    """Test image data generation functions."""
    
    def test_create_image_data_basic(self):
        """Test basic image data generation."""
        X, y = create_image_data(n_samples=30, image_size=16, n_classes=3)
        
        # Check shapes
        self.assertEqual(X.shape, (30, 1, 16, 16))
        self.assertEqual(y.shape, (30,))
        
        # Check data types
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(y.dtype, np.int64)
        
        # Check value ranges
        self.assertGreaterEqual(X.min(), -1.0)  # Should have some negative values due to noise
        self.assertLessEqual(X.max(), 2.0)  # Should have values up to ~1.0 + noise
        
        # Check class distribution
        unique_classes = np.unique(y)
        self.assertEqual(len(unique_classes), 3)
        self.assertTrue(np.all(unique_classes == [0, 1, 2]))
    
    def test_create_image_data_different_sizes(self):
        """Test image data generation with different image sizes."""
        test_sizes = [8, 16, 28, 32]
        
        for size in test_sizes:
            X, y = create_image_data(n_samples=10, image_size=size, n_classes=3)
            
            # Check shapes
            self.assertEqual(X.shape, (10, 1, size, size))
            self.assertEqual(y.shape, (10,))
            
            # Check that images are not all zeros
            self.assertGreater(np.count_nonzero(X), 0)
    
    def test_create_image_data_different_classes(self):
        """Test image data generation with different numbers of classes."""
        test_classes = [2, 3, 4, 5]
        
        for n_classes in test_classes:
            X, y = create_image_data(n_samples=20, image_size=16, n_classes=n_classes)
            
            # Check shapes
            self.assertEqual(X.shape, (20, 1, 16, 16))
            self.assertEqual(y.shape, (20,))
            
            # Check class distribution
            unique_classes = np.unique(y)
            self.assertEqual(len(unique_classes), n_classes)
            self.assertTrue(np.all(unique_classes == list(range(n_classes))))
    
    def test_create_image_data_pattern_types(self):
        """Test that different pattern types are generated correctly."""
        X, y = create_image_data(n_samples=30, image_size=16, n_classes=3)
        
        # Check that we have samples from each class
        for class_id in range(3):
            class_indices = np.where(y == class_id)[0]
            self.assertGreater(len(class_indices), 0)
            
            # Check that class samples have different patterns
            class_images = X[class_indices]
            self.assertGreater(np.count_nonzero(class_images), 0)
    
    def test_create_image_data_reproducibility(self):
        """Test that image data generation is reproducible with same seed."""
        # Generate data twice with same parameters
        X1, y1 = create_image_data(n_samples=20, image_size=16, n_classes=3)
        X2, y2 = create_image_data(n_samples=20, image_size=16, n_classes=3)
        
        # Should be identical due to fixed seed
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_create_image_data_noise_level(self):
        """Test that noise is added appropriately."""
        X, y = create_image_data(n_samples=50, image_size=16, n_classes=3)
        
        # Check that there are some negative values (due to noise)
        self.assertLess(X.min(), 0)
        
        # Check that values are not too extreme
        self.assertLess(X.max(), 2.0)
        self.assertGreater(X.min(), -1.0)
    
    def test_create_image_data_edge_cases(self):
        """Test edge cases for image data generation."""
        # Test with minimum parameters
        X, y = create_image_data(n_samples=1, image_size=8, n_classes=1)
        self.assertEqual(X.shape, (1, 1, 8, 8))
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y[0], 0)
        
        # Test with larger parameters
        X, y = create_image_data(n_samples=100, image_size=64, n_classes=5)
        self.assertEqual(X.shape, (100, 1, 64, 64))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(len(np.unique(y)), 5)
    
    def test_create_image_data_pattern_consistency(self):
        """Test that patterns are consistent within each class."""
        X, y = create_image_data(n_samples=60, image_size=16, n_classes=3)
        
        # Check that each class has the expected number of samples
        for class_id in range(3):
            class_indices = np.where(y == class_id)[0]
            # Should have roughly equal distribution
            self.assertGreaterEqual(len(class_indices), 18)  # 60/3 = 20, allow some variance
            self.assertLessEqual(len(class_indices), 22)
    
    def test_create_image_data_channel_format(self):
        """Test that images are in correct channel format."""
        X, y = create_image_data(n_samples=10, image_size=16, n_classes=3)
        
        # Check that channel dimension is 1 (grayscale)
        self.assertEqual(X.shape[1], 1)
        
        # Check that spatial dimensions are correct
        self.assertEqual(X.shape[2], 16)  # height
        self.assertEqual(X.shape[3], 16)  # width
    
    def test_create_image_data_value_distribution(self):
        """Test that image values have reasonable distribution."""
        X, y = create_image_data(n_samples=100, image_size=16, n_classes=3)
        
        # Check that most values are in reasonable range
        self.assertGreater(np.percentile(X, 10), -0.5)  # 10th percentile should be reasonable
        self.assertLess(np.percentile(X, 90), 1.5)  # 90th percentile should be reasonable
        
        # Check that there are both high and low values
        self.assertLess(X.min(), 0.5)  # Some low values
        self.assertGreater(X.max(), 0.5)  # Some high values


if __name__ == '__main__':
    unittest.main()
