# Import core functionality
from . import mlai

# Import modular components (stub files for refactoring)
from . import models
from . import linear_models
from . import gaussian_processes
from . import neural_networks
from . import utils
from . import dimred
from . import optimisation
from . import experimental
from . import data

# Check for GPy availability
GPY_AVAILABLE = True
try:
    import GPy
except ImportError:
    GPY_AVAILABLE = False

# Import all functions from mlai module for consistent access
from .mlai import *

# Import utility functions to make them available at package level
from .utils import write_figure, write_animation, write_animation_html, filename_join, write_figure_caption, load_pgm, finite_difference_gradient, verify_gradient_implementation, finite_difference_jacobian, dist2
from .data import generate_cluster_data, generate_swiss_roll
from .linear_models import linear, polynomial, radial, fourier, radial_multivariate, Basis, Noise, Gaussian, LM, BLM, LR

# Import neural network classes and functions to make them available at package level
from .neural_networks import (
    SimpleNeuralNetwork, NeuralNetwork,
    init_perceptron, update_perceptron,
    relu_activation, sigmoid_activation, linear_activation, soft_relu_activation,
    LinearActivation, ReLUActivation, SigmoidActivation, SoftReLUActivation,
    LossFunction, MeanSquaredError, MeanAbsoluteError, HuberLoss,
    BinaryCrossEntropyLoss, CrossEntropyLoss
)

# Import plot module
from . import plot

# Import experimental classes to make them available at package level
from .experimental import SimpleDropoutNeuralNetwork, NonparametricDropoutNeuralNetwork

# Import Gaussian Process functions to make them available at package level
from .gaussian_processes import (
    GP, Kernel,
    eq_cov, ou_cov, matern32_cov, matern52_cov, periodic_cov,
    linear_cov, polynomial_cov, relu_cov, bias_cov, add_cov, prod_cov,
    mlp_cov, sinc_cov, brownian_cov, ratquad_cov, basis_cov,
    icm_cov, lmc_cov, slfm_cov,
    posterior_f, update_inverse, contour_data
)

# Import dimensionality reduction and clustering functions to make them available at package level
from .dimred import (
    ClusterModel, WardsMethod,
    kmeans_assignments, kmeans_update, kmeans_objective,
    ppca_eig, ppca_svd, ppca_posterior, kruskal_stress
)

# Import GPy-dependent modules if available
if GPY_AVAILABLE:
    from . import mountain_car
    from . import gp_tutorial
    from . import deepgp_tutorial

# Import transformer components for educational purposes
from .transformer import Attention, MultiHeadAttention, PositionalEncoding, Transformer, SoftmaxActivation, SigmoidAttentionActivation, IdentityMinusSoftmaxActivation
