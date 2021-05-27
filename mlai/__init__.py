from .mlai import *
from . import plot

GPY_AVAILABLE=True
try:
    import GPy
except ImportError:
    GPY_AVAILABLE=False

if GPY_AVAILABLE:
    from . import mountain_car
    from . import gp_tutorial
    from . import deepgp_tutorial
