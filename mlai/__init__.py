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

# Check for GPy availability
GPY_AVAILABLE = True
try:
    import GPy
except ImportError:
    GPY_AVAILABLE = False

# Import all functions from mlai module for consistent access
from .mlai import *

# Import GPy-dependent modules if available
if GPY_AVAILABLE:
    from . import mountain_car
    from . import gp_tutorial
    from . import deepgp_tutorial

# Import transformer components for educational purposes
from .transformer import Attention, MultiHeadAttention, PositionalEncoding, Transformer, SoftmaxActivation, SigmoidAttentionActivation, IdentityMinusSoftmaxActivation
