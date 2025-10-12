# Python code for MLAI lectures.

"""
Machine Learning and Adaptive Intelligence (MLAI) Core Module

This module provides the core machine learning functionality for the MLAI package,
designed for teaching and lecturing on machine learning fundamentals. The module
includes implementations of key algorithms with a focus on clarity and educational value.

Key Components:
-------------
- Linear Models (LM): Basic linear regression with various basis functions
- Bayesian Linear Models (BLM): Linear models with Bayesian inference
- Gaussian Processes (GP): Non-parametric Bayesian models
- Logistic Regression (LR): Binary classification
- Neural Networks: Simple feedforward networks with dropout
- Kernel Functions: Various covariance functions for GPs
- Perceptron: Basic binary classifier for teaching

Mathematical Focus:
------------------
The implementations emphasize mathematical transparency, with clear connections
between code and mathematical notation. Each class and function includes
mathematical explanations where relevant.

Educational Design:
------------------
- Simple, readable implementations suitable for teaching
- Clear separation of concepts
- Extensive use of mathematical notation in variable names
- Comprehensive examples and documentation

For detailed usage examples, see the tutorials in gp_tutorial.py, deepgp_tutorial.py,
and mountain_car.py.

Author: Neil D. Lawrence
License: MIT
"""

# import the time model to allow python to pause.
import time
import os
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

import numpy as np
import scipy.linalg as la

from numpy import vstack

# Import linear models from the linear_models module
from .linear_models import LM
from .utils import dist2
from .models import Model, ProbModel, MapModel, ProbMapModel










##########          Week 8            ##########

    

# Code for loading pgm from http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
def load_pgm(filename, directory=None, byteorder='>'):
    """
    Return image data from a raw PGM file as a numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    :param filename: Name of the PGM file to load
    :type filename: str
    :param directory: Directory containing the file (optional)
    :type directory: str, optional
    :param byteorder: Byte order for 16-bit images ('>' for big-endian, '<' for little-endian)
    :type byteorder: str, optional
    :returns: Image data as a numpy array
    :rtype: numpy.ndarray
    :raises ValueError: If the file is not a valid raw PGM file
    """
    import re
    from .utils import filename_join
    savename = filename_join(filename, directory)
    with open(savename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(
        buffer,
        dtype='u1' if int(maxval) < 256 else byteorder+'u2',
        count=int(width)*int(height),
        offset=len(header)
    ).reshape((int(height), int(width)))


