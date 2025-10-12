import unittest
import numpy as np
from mlai.neural_networks import (
    ConvolutionalLayer, MaxPoolingLayer, FlattenLayer, 
    ReLUActivation, LinearActivation, LayeredNeuralNetwork, LinearLayer
)
from mlai.utils import finite_difference_gradient, verify_gradient_implementation


class TestConvolutionalLayer(unittest.TestCase):
    """Test ConvolutionalLayer implementation."""
    
    def test_convolutional_layer_creation(self):
        """Test convolutional layer creation and basic properties."""
        conv = ConvolutionalLayer(input_channels=3, output_channels=16, kernel_size=3)
        
        self.assertEqual(conv.input_channels, 3)
        self.assertEqual(conv.output_channels, 16)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (0, 0))
        self.assertEqual(conv.filters.shape, (16, 3, 3, 3))
        self.assertEqual(conv.biases.shape, (16,))
        self.assertIsInstance(conv.activation, ReLUActivation)
    
    def test_convolutional_layer_forward_pass(self):
        """Test convolutional layer forward pass."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=2, kernel_size=3, stride=1, padding=0)
        
        # Test input: batch_size=2, channels=1, height=5, width=5
        x = np.random.randn(2, 1, 5, 5)
        output = conv.forward(x)
        
        # Expected output shape: (2, 2, 3, 3) for 5x5 input with 3x3 kernel, stride=1, padding=0
        expected_height = (5 - 3) // 1 + 1  # = 3
        expected_width = (5 - 3) // 1 + 1   # = 3
        self.assertEqual(output.shape, (2, 2, expected_height, expected_width))
        
        # Check that output is not all zeros (convolution worked)
        self.assertFalse(np.allclose(output, 0))
    
    def test_convolutional_layer_with_padding(self):
        """Test convolutional layer with padding."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=2, kernel_size=3, padding=1)
        
        x = np.random.randn(1, 1, 5, 5)
        output = conv.forward(x)
        
        # With padding=1, output should be same size as input
        self.assertEqual(output.shape, (1, 2, 5, 5))
    
    def test_convolutional_layer_with_stride(self):
        """Test convolutional layer with stride."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=2, kernel_size=3, stride=2)
        
        x = np.random.randn(1, 1, 7, 7)
        output = conv.forward(x)
        
        # With stride=2, output should be roughly half the size
        expected_height = (7 - 3) // 2 + 1  # = 3
        expected_width = (7 - 3) // 2 + 1   # = 3
        self.assertEqual(output.shape, (1, 2, expected_height, expected_width))
    
    def test_convolutional_layer_parameters(self):
        """Test convolutional layer parameter management."""
        conv = ConvolutionalLayer(input_channels=2, output_channels=3, kernel_size=2)
        
        # Test getting parameters
        params = conv.parameters
        expected_size = conv.filters.size + conv.biases.size
        self.assertEqual(len(params), expected_size)
        
        # Test setting parameters
        new_params = np.random.randn(len(params))
        conv.parameters = new_params
        np.testing.assert_array_equal(conv.parameters, new_params)
    
    def test_convolutional_layer_gradient_consistency(self):
        """Test that forward and backward passes are consistent."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=2, kernel_size=3)
        
        x = np.random.randn(1, 1, 5, 5)
        
        # Forward pass
        output = conv.forward(x)
        
        # Backward pass with dummy gradient
        grad_output = np.ones_like(output)
        grad_input = conv.backward(grad_output)
        
        # Check that gradient has correct shape
        self.assertEqual(grad_input[0].shape, x.shape)
        
        # Check that gradient is not all zeros
        self.assertFalse(np.allclose(grad_input[0], 0))
    
    def test_convolutional_layer_gradient_verification(self):
        """Test convolutional layer gradients using finite differences."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=1, kernel_size=2)
        
        x = np.random.randn(1, 1, 4, 4)
        
        # Test input gradient
        def conv_func(x_flat):
            x_reshaped = x_flat.reshape(1, 1, 4, 4)
            output = conv.forward(x_reshaped)
            return np.sum(output)
        
        # Numerical gradient
        numerical_grad = finite_difference_gradient(conv_func, x.flatten())
        
        # Analytical gradient
        output = conv.forward(x)
        analytical_grad_tuple = conv.backward(np.ones_like(output))
        analytical_grad = analytical_grad_tuple[0].flatten()
        
        # Verify gradients match
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad, rtol=1e-4))
    
    def test_convolutional_layer_different_architectures(self):
        """Test convolutional layer with different architectures."""
        test_cases = [
            (1, 1, 2, 1, 0),  # Single channel, single filter
            (3, 4, 3, 1, 1),  # RGB input, multiple filters, padding
            (2, 8, 5, 2, 0),  # Multiple channels, large kernel, stride
        ]
        
        for input_channels, output_channels, kernel_size, stride, padding in test_cases:
            conv = ConvolutionalLayer(input_channels, output_channels, kernel_size, stride, padding)
            
            # Create test input
            x = np.random.randn(2, input_channels, 8, 8)
            
            # Forward pass
            output = conv.forward(x)
            
            # Check output shape
            expected_height = (8 + 2 * padding - kernel_size) // stride + 1
            expected_width = (8 + 2 * padding - kernel_size) // stride + 1
            self.assertEqual(output.shape, (2, output_channels, expected_height, expected_width))
            
            # Backward pass
            grad_output = np.ones_like(output)
            grad_input = conv.backward(grad_output)
            self.assertEqual(grad_input[0].shape, x.shape)


class TestMaxPoolingLayer(unittest.TestCase):
    """Test MaxPoolingLayer implementation."""
    
    def test_max_pooling_layer_creation(self):
        """Test max pooling layer creation."""
        pool = MaxPoolingLayer(pool_size=2)
        
        self.assertEqual(pool.pool_size, (2, 2))
        self.assertEqual(pool.stride, (2, 2))
        self.assertEqual(len(pool.parameters), 0)  # No trainable parameters
    
    def test_max_pooling_layer_forward_pass(self):
        """Test max pooling layer forward pass."""
        pool = MaxPoolingLayer(pool_size=2, stride=2)
        
        # Test input: batch_size=1, channels=2, height=4, width=4
        x = np.random.randn(1, 2, 4, 4)
        output = pool.forward(x)
        
        # Expected output shape: (1, 2, 2, 2) for 4x4 input with 2x2 pool, stride=2
        self.assertEqual(output.shape, (1, 2, 2, 2))
        
        # Check that output values are from input (max pooling property)
        for b in range(1):
            for c in range(2):
                for h in range(2):
                    for w in range(2):
                        # Get corresponding input region
                        start_h, start_w = h * 2, w * 2
                        region = x[b, c, start_h:start_h+2, start_w:start_w+2]
                        expected_max = np.max(region)
                        self.assertEqual(output[b, c, h, w], expected_max)
    
    def test_max_pooling_layer_gradient_consistency(self):
        """Test that max pooling forward and backward passes are consistent."""
        pool = MaxPoolingLayer(pool_size=2, stride=2)
        
        x = np.random.randn(1, 1, 4, 4)
        
        # Forward pass
        output = pool.forward(x)
        
        # Backward pass
        grad_output = np.ones_like(output)
        grad_input = pool.backward(grad_output)
        
        # Check gradient shape
        self.assertEqual(grad_input[0].shape, x.shape)
        
        # Check that gradients are only at maximum positions
        # (gradients should be 1.0 at max positions, 0.0 elsewhere)
        for b in range(1):
            for c in range(1):
                for h in range(2):
                    for w in range(2):
                        # Find the maximum position in original input
                        start_h, start_w = h * 2, w * 2
                        region = x[b, c, start_h:start_h+2, start_w:start_w+2]
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        max_h, max_w = start_h + max_idx[0], start_w + max_idx[1]
                        
                        # Gradient should be 1.0 at max position
                        self.assertEqual(grad_input[0][b, c, max_h, max_w], 1.0)
    
    def test_max_pooling_layer_different_sizes(self):
        """Test max pooling with different pool sizes."""
        test_cases = [
            (2, 2),  # Standard 2x2 pooling
            (3, 3),  # 3x3 pooling
            (2, 1),  # Non-square pooling
        ]
        
        for pool_h, pool_w in test_cases:
            pool = MaxPoolingLayer(pool_size=(pool_h, pool_w))
            
            # Create test input
            x = np.random.randn(1, 1, 6, 6)
            
            # Forward pass
            output = pool.forward(x)
            
            # Check output shape
            expected_h = (6 - pool_h) // pool_h + 1
            expected_w = (6 - pool_w) // pool_w + 1
            self.assertEqual(output.shape, (1, 1, expected_h, expected_w))


class TestFlattenLayer(unittest.TestCase):
    """Test FlattenLayer implementation."""
    
    def test_flatten_layer_creation(self):
        """Test flatten layer creation."""
        flatten = FlattenLayer()
        
        self.assertEqual(flatten.start_dim, 1)
        self.assertEqual(flatten.end_dim, -1)
        self.assertEqual(len(flatten.parameters), 0)  # No trainable parameters
    
    def test_flatten_layer_forward_pass(self):
        """Test flatten layer forward pass."""
        flatten = FlattenLayer()
        
        # Test input: batch_size=2, channels=3, height=4, width=4
        x = np.random.randn(2, 3, 4, 4)
        output = flatten.forward(x)
        
        # Expected output shape: (2, 48) - batch_size preserved, rest flattened
        expected_size = 3 * 4 * 4  # channels * height * width
        self.assertEqual(output.shape, (2, expected_size))
        
        # Check that values are preserved
        np.testing.assert_array_equal(output, x.reshape(2, -1))
    
    def test_flatten_layer_gradient_consistency(self):
        """Test that flatten forward and backward passes are consistent."""
        flatten = FlattenLayer()
        
        x = np.random.randn(2, 3, 4, 4)
        
        # Forward pass
        output = flatten.forward(x)
        
        # Backward pass
        grad_output = np.ones_like(output)
        grad_input = flatten.backward(grad_output)
        
        # Check gradient shape
        self.assertEqual(grad_input[0].shape, x.shape)
        
        # Check that gradient values are preserved
        np.testing.assert_array_equal(grad_input[0], grad_output.reshape(x.shape))
    
    def test_flatten_layer_custom_dimensions(self):
        """Test flatten layer with custom start/end dimensions."""
        # Test custom flattening
        flatten = FlattenLayer(start_dim=1, end_dim=2)
        
        x = np.random.randn(2, 3, 4, 5)
        output = flatten.forward(x)
        
        # Should flatten dimensions 1 and 2 (channels and height)
        expected_size = 3 * 4  # channels * height
        self.assertEqual(output.shape, (2, expected_size, 5))


class TestConvolutionalLayeredNetwork(unittest.TestCase):
    """Test convolutional layers in a layered network."""
    
    def test_convolutional_layered_network(self):
        """Test convolutional layers in a layered neural network."""
        # Create a simple CNN architecture
        layers = [
            ConvolutionalLayer(input_channels=1, output_channels=4, kernel_size=3, padding=1),
            MaxPoolingLayer(pool_size=2, stride=2),
            ConvolutionalLayer(input_channels=4, output_channels=8, kernel_size=3, padding=1),
            MaxPoolingLayer(pool_size=2, stride=2),
            FlattenLayer(),
            LinearLayer(8 * 2 * 2, 10),  # 8 channels * 2*2 spatial after pooling
        ]
        
        # Create layered network
        cnn = LayeredNeuralNetwork(layers)
        
        # Test input: batch_size=2, channels=1, height=8, width=8
        x = np.random.randn(2, 1, 8, 8)
        
        # Forward pass
        output = cnn.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10))
        
        # Check that network has parameters
        self.assertGreater(len(cnn.parameters), 0)
    
    def test_convolutional_network_gradient_verification(self):
        """Test convolutional network gradients using finite differences."""
        # Create a simple CNN
        layers = [
            ConvolutionalLayer(input_channels=1, output_channels=2, kernel_size=2),
            FlattenLayer(),
            LinearLayer(2 * 3 * 3, 1),  # 2 channels * 3*3 spatial
        ]
        
        cnn = LayeredNeuralNetwork(layers)
        
        x = np.random.randn(1, 1, 4, 4)
        
        # Test input gradient
        def cnn_func(x_flat):
            x_reshaped = x_flat.reshape(1, 1, 4, 4)
            output = cnn.forward(x_reshaped)
            return np.sum(output)
        
        # Numerical gradient
        numerical_grad = finite_difference_gradient(cnn_func, x.flatten())
        
        # Analytical gradient
        output = cnn.forward(x)
        analytical_grad_dict = cnn.backward(np.ones_like(output))
        # Get the gradient from the first layer (input gradient)
        analytical_grad = analytical_grad_dict['layer_0'][0].flatten()
        
        # Verify gradients match
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad, rtol=1e-4))


class TestConvolutionalLayerEdgeCases(unittest.TestCase):
    """Test convolutional layer edge cases and error handling."""
    
    def test_convolutional_layer_single_pixel(self):
        """Test convolutional layer with very small input."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=1, kernel_size=1)
        
        x = np.random.randn(1, 1, 1, 1)
        output = conv.forward(x)
        
        self.assertEqual(output.shape, (1, 1, 1, 1))
    
    def test_convolutional_layer_large_kernel(self):
        """Test convolutional layer with kernel larger than input."""
        conv = ConvolutionalLayer(input_channels=1, output_channels=1, kernel_size=5, padding=2)
        
        x = np.random.randn(1, 1, 3, 3)
        output = conv.forward(x)
        
        # With padding=2, should maintain size
        self.assertEqual(output.shape, (1, 1, 3, 3))
    
    def test_convolutional_layer_parameter_consistency(self):
        """Test that parameter getter/setter maintains consistency."""
        conv = ConvolutionalLayer(input_channels=2, output_channels=3, kernel_size=2)
        
        # Get original parameters
        original_params = conv.parameters.copy()
        
        # Modify parameters
        new_params = original_params + 0.1
        conv.parameters = new_params
        
        # Check that parameters were set correctly
        np.testing.assert_array_almost_equal(conv.parameters, new_params)
        
        # Check that filters and biases were updated correctly
        filters_size = conv.filters.size
        np.testing.assert_array_almost_equal(conv.filters.flatten(), new_params[:filters_size])
        np.testing.assert_array_almost_equal(conv.biases, new_params[filters_size:])


if __name__ == '__main__':
    unittest.main()
