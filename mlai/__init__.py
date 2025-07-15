# Import core functionality
from . import mlai

# Check for GPy availability
GPY_AVAILABLE = True
try:
    import GPy
except ImportError:
    GPY_AVAILABLE = False

# Import GPy-dependent modules if available
if GPY_AVAILABLE:
    from . import mountain_car
    from . import gp_tutorial
    from . import deepgp_tutorial

# Only expose core utilities and Basis at the package level
from .mlai import write_figure, write_animation

# Import all functions from mlai module for consistent access
from .mlai import *
