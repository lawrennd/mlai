# Import core functionality
from . import mlai

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
