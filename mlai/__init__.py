from . import mlai
from . import plot

gpy_available=True
try:
    import GPy
except ImportError:
    gpy_available=False

if gpy_available:
    from . import mountain_car
    from . import gp_tutorial
    from . import deepgp_tutorial
