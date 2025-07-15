# Import core functionality
from . import mlai
from . import plot

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

# Make key functions available at package level
from .mlai import write_figure, write_animation, Basis, linear, icm_cov, slfm_cov
