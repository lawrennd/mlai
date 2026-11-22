"""
Finite-difference checks of parameter and input gradients for the layered
architecture (linear, convolutional, pooling, attention layers and
LayeredNeuralNetwork).

Each test compares the analytical gradient from ``backward()`` /
``gradients`` with a central finite-difference estimate of the gradient of
a random linear functional of the layer output, L = sum(G * output).
"""
import numpy as np
import pytest

from mlai.neural_networks import (
    LinearLayer, FullyConnectedLayer, ConvolutionalLayer, MaxPoolingLayer,
    FlattenLayer, AttentionLayer, MultiHeadAttentionLayer, PositionalEncodingLayer,
    LayeredNeuralNetwork, ReLUActivation, SigmoidActivation, SoftReLUActivation,
)
from mlai.loss import MeanSquaredError, SoftmaxCrossEntropyLoss

RTOL = 1e-5
ATOL = 1e-7


def _numerical_gradient(f, x, h=1e-6):
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += h
        xm.flat[i] -= h
        grad.flat[i] = (f(xp) - f(xm)) / (2 * h)
    return grad


def _check_layer(layer, x, forward=None, seed=None):
    """Check input and parameter gradients of a single-input layer."""
    forward = forward or (lambda z: layer.forward(z))

    def run(z):
        if seed is not None:
            np.random.seed(seed)
        return forward(z)

    G = np.random.randn(*run(x).shape)

    # Input gradient
    numerical_x = _numerical_gradient(lambda z: np.sum(run(z) * G), x)
    run(x)
    analytical_x = layer.backward(G)
    if isinstance(analytical_x, tuple):
        analytical_x = analytical_x[0]
    np.testing.assert_allclose(analytical_x, numerical_x, rtol=RTOL, atol=ATOL)

    # Parameter gradient
    p0 = layer.parameters.copy()
    if p0.size == 0:
        assert layer.gradients.size == 0
        return

    def loss_of_params(p):
        layer.parameters = p
        return np.sum(run(x) * G)

    numerical_p = _numerical_gradient(loss_of_params, p0)
    layer.parameters = p0
    run(x)
    layer.backward(G)
    np.testing.assert_allclose(layer.gradients, numerical_p, rtol=RTOL, atol=ATOL)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


class TestLinearLayers:
    def test_linear_layer_2d(self):
        _check_layer(LinearLayer(3, 4), np.random.randn(5, 3))

    def test_linear_layer_sequence_input(self):
        _check_layer(LinearLayer(3, 4), np.random.randn(2, 5, 3))

    def test_fully_connected_layer(self):
        _check_layer(FullyConnectedLayer(3, 4, SigmoidActivation()), np.random.randn(5, 3))

    def test_gradients_before_backward_raises(self):
        with pytest.raises(ValueError):
            _ = LinearLayer(2, 2).gradients


class TestConvolutionalLayer:
    @pytest.mark.parametrize("padding,stride", [(0, 1), (1, 1), (1, 2), (2, 1), ((1, 0), 1), ((0, 2), (2, 1))])
    def test_input_and_filter_gradients(self, padding, stride):
        layer = ConvolutionalLayer(2, 3, 3, stride=stride, padding=padding, activation=SoftReLUActivation())
        _check_layer(layer, np.random.randn(2, 2, 6, 6))

    def test_asymmetric_padding_shapes(self):
        layer = ConvolutionalLayer(1, 1, 3, padding=(1, 0))
        x = np.random.randn(1, 1, 5, 5)
        out = layer.forward(x)
        assert out.shape == (1, 1, 5, 3)
        assert layer.backward(np.ones_like(out))[0].shape == x.shape


class TestParameterFreeLayers:
    def test_max_pooling(self):
        _check_layer(MaxPoolingLayer(pool_size=2, stride=2), np.random.randn(2, 2, 6, 6))

    def test_flatten(self):
        _check_layer(FlattenLayer(), np.random.randn(2, 2, 3, 3))

    def test_positional_encoding(self):
        _check_layer(PositionalEncodingLayer(d_model=4), np.random.randn(2, 3, 4))


class TestAttentionLayer:
    def test_self_attention(self):
        _check_layer(AttentionLayer(4), np.random.randn(2, 3, 4))

    def test_self_attention_smaller_head_dimension(self):
        _check_layer(AttentionLayer(4, d_k=2, output_projection=False), np.random.randn(2, 3, 4))

    def test_causal_self_attention(self):
        _check_layer(AttentionLayer(4, causal=True), np.random.randn(2, 4, 4))

    def test_causal_mask_blocks_future(self):
        layer = AttentionLayer(4, causal=True)
        x = np.random.randn(1, 5, 4)
        out = layer.forward(x)
        x2 = x.copy()
        x2[:, 3:] += 1.0  # change only the last two positions
        out2 = layer.forward(x2)
        np.testing.assert_allclose(out[:, :3], out2[:, :3])
        assert np.allclose(np.triu(layer.last_attention_weights[0], k=1), 0.0)

    def test_elementwise_activation_before_softmax(self):
        _check_layer(AttentionLayer(4, activation=SoftReLUActivation()), np.random.randn(2, 3, 4))

    def test_dropout(self):
        layer = AttentionLayer(4, dropout=0.5)
        _check_layer(layer, np.random.randn(2, 3, 4), seed=7)

    @pytest.mark.parametrize("mode", ["cross", "mixed"])
    def test_cross_and_mixed_attention(self, mode):
        layer = AttentionLayer(4)
        q = np.random.randn(2, 3, 4)
        kv = np.random.randn(2, 5, 4)

        def forward(qq, kk):
            if mode == "cross":
                return layer.forward(None, query_input=qq, key_value_input=kk)
            return layer.forward(qq, key_value_input=kk)

        G = np.random.randn(*forward(q, kv).shape)
        num_q = _numerical_gradient(lambda z: np.sum(forward(z, kv) * G), q)
        num_kv = _numerical_gradient(lambda z: np.sum(forward(q, z) * G), kv)
        forward(q, kv)
        grad_q, grad_kv = layer.backward(G)
        np.testing.assert_allclose(grad_q, num_q, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(grad_kv, num_kv, rtol=RTOL, atol=ATOL)

        p0 = layer.parameters.copy()

        def loss_of_params(p):
            layer.parameters = p
            return np.sum(forward(q, kv) * G)

        num_p = _numerical_gradient(loss_of_params, p0)
        layer.parameters = p0
        forward(q, kv)
        layer.backward(G)
        np.testing.assert_allclose(layer.gradients, num_p, rtol=RTOL, atol=ATOL)


class TestMultiHeadAttentionLayer:
    def test_gradients(self):
        _check_layer(MultiHeadAttentionLayer(6, 2), np.random.randn(2, 3, 6))

    def test_causal_gradients(self):
        _check_layer(MultiHeadAttentionLayer(6, 3, causal=True), np.random.randn(2, 4, 6))

    def test_every_head_sees_every_feature(self):
        mha = MultiHeadAttentionLayer(8, 2)
        x = np.random.randn(1, 4, 8)
        mha.forward(x)
        head0 = mha.last_head_outputs[0].copy()
        x2 = x.copy()
        x2[..., 4:] += 1.0  # features that the old implementation hid from head 0
        mha.forward(x2)
        assert not np.allclose(head0, mha.last_head_outputs[0])

    def test_attention_weights_shape(self):
        mha = MultiHeadAttentionLayer(8, 2)
        mha.forward(np.random.randn(3, 5, 8))
        assert mha.last_attention_weights.shape == (3, 2, 5, 5)


class TestLayeredNeuralNetworkGradients:
    def _check_network(self, network, x, y, loss):
        p0 = network.parameters.copy()

        def loss_of_params(p):
            network.parameters = p
            return loss.forward(network.forward(x), y)

        numerical = _numerical_gradient(loss_of_params, p0)
        network.parameters = p0
        out = network.forward(x)
        network.set_output_gradient(loss.gradient(out, y))
        np.testing.assert_allclose(network.gradients, numerical, rtol=1e-4, atol=1e-7)

    def test_cnn(self):
        network = LayeredNeuralNetwork([
            ConvolutionalLayer(1, 2, 3, padding=1, activation=SoftReLUActivation()),
            MaxPoolingLayer(pool_size=2),
            FlattenLayer(),
            FullyConnectedLayer(2 * 3 * 3, 4, SigmoidActivation()),
            LinearLayer(4, 3),
        ])
        x = np.random.randn(2, 1, 6, 6)
        y = np.eye(3)[[0, 2]]
        self._check_network(network, x, y, SoftmaxCrossEntropyLoss())

    def test_transformer(self):
        vocab, d_model, seq_len = 5, 4, 3
        network = LayeredNeuralNetwork([
            LinearLayer(vocab, d_model),           # embedding of one-hot tokens
            PositionalEncodingLayer(d_model),
            MultiHeadAttentionLayer(d_model, 2, causal=True),
            LinearLayer(d_model, vocab),
        ])
        tokens = np.random.randint(vocab, size=(2, seq_len))
        x = np.eye(vocab)[tokens]
        y = np.eye(vocab)[np.roll(tokens, -1, axis=1)]
        self._check_network(network, x, y, SoftmaxCrossEntropyLoss())

    def test_mse_regression_network(self):
        network = LayeredNeuralNetwork([
            FullyConnectedLayer(3, 4, ReLUActivation()),
            LinearLayer(4, 2),
        ])
        self._check_network(network, np.random.randn(5, 3), np.random.randn(5, 2), MeanSquaredError())


class TestSoftmaxCrossEntropyLoss:
    @pytest.mark.parametrize("shape", [(4, 3), (2, 5, 3)])
    def test_gradient(self, shape):
        loss = SoftmaxCrossEntropyLoss()
        logits = np.random.randn(*shape)
        targets = np.eye(shape[-1])[np.random.randint(shape[-1], size=shape[:-1])]
        numerical = _numerical_gradient(lambda z: loss.forward(z, targets), logits)
        np.testing.assert_allclose(loss.gradient(logits, targets), numerical, rtol=RTOL, atol=ATOL)

    def test_matches_cross_entropy_on_probabilities(self):
        from mlai.loss import CrossEntropyLoss
        logits = np.random.randn(4, 3)
        targets = np.eye(3)[[0, 1, 2, 1]]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert np.isclose(SoftmaxCrossEntropyLoss().forward(logits, targets),
                          CrossEntropyLoss().forward(probs, targets))

    def test_large_logits_are_stable(self):
        loss = SoftmaxCrossEntropyLoss()
        logits = np.array([[1000.0, -1000.0]])
        assert np.isfinite(loss.forward(logits, np.array([[1.0, 0.0]])))
