# Package version
__version__ = "0.1.3"

# Import modular components
from . import models
from . import linear_models
from . import gaussian_processes
from . import neural_networks
from . import utils
from . import dimred
from . import optimisation
from . import hmc
from . import experimental
from . import data
from . import loss
from . import plot

from .models import Model, ProbModel, MapModel, ProbMapModel
from .utils import (
    write_figure, write_animation, write_animation_html, filename_join,
    write_figure_caption, load_pgm, finite_difference_gradient,
    verify_gradient_implementation, finite_difference_jacobian, dist2,
)
from .data import generate_cluster_data, generate_swiss_roll, create_image_data
from .linear_models import (
    linear, polynomial, radial, fourier, radial_multivariate,
    relu, hyperbolic_tangent,
    Basis, Noise, Gaussian, LM, BLM, LR,
)
from .neural_networks import (
    SimpleNeuralNetwork, NeuralNetwork, LayeredNeuralNetwork,
    init_perceptron, update_perceptron,
    relu_activation, sigmoid_activation, linear_activation, soft_relu_activation,
    Activation, LinearActivation, ReLUActivation, SigmoidActivation, SoftReLUActivation,
    Layer, LinearLayer, FullyConnectedLayer, MultiInputLayer,
    AttentionLayer, MultiHeadAttentionLayer, PositionalEncodingLayer,
    ConvolutionalLayer, MaxPoolingLayer, FlattenLayer,
)
from .loss import (
    LossFunction, MeanSquaredError, MeanAbsoluteError, HuberLoss,
    BinaryCrossEntropyLoss, CrossEntropyLoss,
)
from .experimental import SimpleDropoutNeuralNetwork, NonparametricDropoutNeuralNetwork
from .gaussian_processes import (
    GP, Kernel,
    eq_cov, exponentiated_quadratic, ou_cov, matern32_cov, matern52_cov, periodic_cov,
    linear_cov, polynomial_cov, relu_cov, bias_cov, add_cov, prod_cov,
    arccos_cov, mlp_cov, sinc_cov, brownian_cov, ratquad_cov, basis_cov,
    icm_cov, lmc_cov, slfm_cov,
    posterior_f, update_inverse, contour_data,
)
from .dimred import (
    ClusterModel, WardsMethod,
    kmeans_assignments, kmeans_update, kmeans_objective,
    ppca_eig, ppca_svd, ppca_posterior, kruskal_stress,
)
from .optimisation import Optimiser, SGD, Adam, train_model
from .hmc import HamiltonianMonteCarlo, HMCResult, leapfrog, kinetic_energy

try:
    import GPy
except ImportError:
    GPy = None
GPY_AVAILABLE = GPy is not None

__all__ = [
    "__version__",
    "models",
    "linear_models",
    "gaussian_processes",
    "neural_networks",
    "utils",
    "dimred",
    "optimisation",
    "hmc",
    "experimental",
    "data",
    "loss",
    "plot",
    "Model",
    "ProbModel",
    "MapModel",
    "ProbMapModel",
    "write_figure",
    "write_animation",
    "write_animation_html",
    "filename_join",
    "write_figure_caption",
    "load_pgm",
    "finite_difference_gradient",
    "verify_gradient_implementation",
    "finite_difference_jacobian",
    "dist2",
    "generate_cluster_data",
    "generate_swiss_roll",
    "create_image_data",
    "linear",
    "polynomial",
    "radial",
    "fourier",
    "radial_multivariate",
    "relu",
    "hyperbolic_tangent",
    "Basis",
    "Noise",
    "Gaussian",
    "LM",
    "BLM",
    "LR",
    "SimpleNeuralNetwork",
    "NeuralNetwork",
    "LayeredNeuralNetwork",
    "init_perceptron",
    "update_perceptron",
    "relu_activation",
    "sigmoid_activation",
    "linear_activation",
    "soft_relu_activation",
    "Activation",
    "LinearActivation",
    "ReLUActivation",
    "SigmoidActivation",
    "SoftReLUActivation",
    "Layer",
    "LinearLayer",
    "FullyConnectedLayer",
    "MultiInputLayer",
    "AttentionLayer",
    "MultiHeadAttentionLayer",
    "PositionalEncodingLayer",
    "ConvolutionalLayer",
    "MaxPoolingLayer",
    "FlattenLayer",
    "LossFunction",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "HuberLoss",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "SimpleDropoutNeuralNetwork",
    "NonparametricDropoutNeuralNetwork",
    "GP",
    "Kernel",
    "eq_cov",
    "exponentiated_quadratic",
    "ou_cov",
    "matern32_cov",
    "matern52_cov",
    "periodic_cov",
    "linear_cov",
    "polynomial_cov",
    "relu_cov",
    "bias_cov",
    "add_cov",
    "prod_cov",
    "arccos_cov",
    "mlp_cov",
    "sinc_cov",
    "brownian_cov",
    "ratquad_cov",
    "basis_cov",
    "icm_cov",
    "lmc_cov",
    "slfm_cov",
    "posterior_f",
    "update_inverse",
    "contour_data",
    "ClusterModel",
    "WardsMethod",
    "kmeans_assignments",
    "kmeans_update",
    "kmeans_objective",
    "ppca_eig",
    "ppca_svd",
    "ppca_posterior",
    "kruskal_stress",
    "Optimiser",
    "SGD",
    "Adam",
    "train_model",
    "HamiltonianMonteCarlo",
    "HMCResult",
    "leapfrog",
    "kinetic_energy",
    "GPy",
    "GPY_AVAILABLE",
]

# gp_tutorial is pure matplotlib/numpy — always export it.
from . import gp_tutorial  # noqa: E402
__all__ += ["gp_tutorial"]

# GPy-dependent modules stay out of the unconditional import list so
# `import mlai` succeeds when GPy is absent.
if GPY_AVAILABLE:
    from . import mountain_car  # noqa: E402
    from . import deepgp_tutorial  # noqa: E402
    __all__ += ["mountain_car", "deepgp_tutorial"]
