"""
Utilities Module

This module contains utility functions including:
- Optimization utilities (finite_difference_gradient, finite_difference_jacobian, verify_gradient_implementation)
- Data generation utilities (load_pgm, generate_swiss_roll)
- Mathematical utilities (radial_multivariate, dist2)
- Plotting utilities (write_figure_caption)

TODO: Extract from mlai.py during refactoring
"""

import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    # Optimization Utilities
    'finite_difference_gradient',
    'finite_difference_jacobian', 
    'verify_gradient_implementation',
    
    # Data Generation
    'load_pgm',
    'generate_swiss_roll',
    
    # Mathematical Utilities
    'radial_multivariate',
    'dist2',
    
    # Plotting Utilities
    'write_figure_caption',
]

def filename_join(filename, directory=None):
    """
    Join a filename to a directory and create directory if it doesn't exist.
    
    This utility function ensures that the target directory exists before
    attempting to create files, which is useful for saving figures, animations,
    and other outputs during tutorials.
    
    :param filename: The name of the file to create
    :type filename: str
    :param directory: The directory path. If None, returns just the filename.
        If the directory doesn't exist, it will be created.
    :type directory: str, optional
    :returns: The full path to the file
    :rtype: str
    
    Examples:
        >>> filename_join("plot.png", "figures")
        'figures/plot.png'
        >>> filename_join("data.csv")
        'data.csv'
    """
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    return filename

def write_animation(anim, filename, directory=None, **kwargs):
    """
    Write a matplotlib animation to a file.
    
    This function saves animations (e.g., from matplotlib.animation) to files
    in the specified directory, creating the directory if necessary.
    
    :param anim: The animation object to save
    :type anim: matplotlib.animation.Animation
    :param filename: The name of the output file (e.g., 'animation.gif', 'animation.mp4')
    :type filename: str
    :param directory: The directory to save the animation in. If None, saves in current directory.
    :type directory: str, optional
    :param **kwargs: Additional arguments passed to anim.save()
    :type **kwargs: dict
    
    Examples:
        >>> import matplotlib.animation as animation
        >>> # Create animation...
        >>> write_animation(anim, "learning_process.gif", "animations")
    """
    savename = filename_join(filename, directory)
    anim.save(savename, **kwargs)

def write_animation_html(anim, filename, directory=None):
    """
    Save a matplotlib animation as an HTML file with embedded JavaScript.
    
    This function creates an HTML file containing the animation that can be
    viewed in a web browser. The animation is embedded as JavaScript code.
    
    :param anim: The animation object to save
    :type anim: matplotlib.animation.Animation
    :param filename: The name of the output HTML file
    :type filename: str
    :param directory: The directory to save the HTML file in. If None, saves in current directory.
    :type directory: str, optional
    
    Examples:
        >>> import matplotlib.animation as animation
        >>> # Create animation...
        >>> write_animation_html(anim, "learning_process.html", "animations")
    """
    savename = filename_join(filename, directory)
    f = open(savename, 'w')
    f.write(anim.to_jshtml())
    f.close()

def write_figure(filename, figure=None, directory=None, frameon=None, **kwargs):
    """
    Save a matplotlib figure to a file with proper formatting.
    
    This function saves figures with transparent background by default,
    which is useful for presentations and publications. The function
    automatically creates the target directory if it doesn't exist.
    
    :param filename: The name of the output file (e.g., 'plot.png', 'figure.pdf')
    :type filename: str
    :param figure: The figure to save. If None, saves the current figure.
    :type figure: matplotlib.figure.Figure, optional
    :param directory: The directory to save the figure in. If None, saves in current directory.
    :type directory: str, optional
    :param frameon: Whether to draw a frame around the figure. If None, uses matplotlib default.
    :type frameon: bool, optional
    :param **kwargs: Additional arguments passed to plt.savefig() or figure.savefig()
    :type **kwargs: dict
    
    Examples:
        >>> plt.plot([1, 2, 3], [1, 4, 2])
        >>> write_figure("linear_plot.png", directory="figures")
        >>> write_figure("presentation_plot.png", transparent=False, dpi=300)
    """
    savename = filename_join(filename, directory)
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    if figure is None:
        plt.savefig(savename, **kwargs)
    else:
        figure.savefig(savename, **kwargs)

def write_figure_caption(counter, caption, filestub, ext="svg", directory="./diagrams", frameon=None, **kwargs):
    """Helper function to save plots with captions for animation"""

    write_figure(f"{filestub}_{counter:0>3}.{ext}", directory=directory, frameon=frameon, **kwargs)
    with open(os.path.join(directory,f"{filestub}_{counter:0>3}.md"), 'w') as f:
        f.write(caption)

        
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
    import numpy as np
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




def finite_difference_gradient(func, x, h=1e-5):
    """
    Compute gradient using finite differences.
    
    This is a numerical method to approximate gradients by computing
    the difference quotient: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    This is useful for:
    1. Verifying analytical gradient implementations
    2. Educational purposes to understand gradient computation
    3. Debugging gradient-related issues
    
    :param func: Function to compute gradient for
    :type func: callable
    :param x: Point at which to compute gradient
    :type x: numpy.ndarray
    :param h: Step size for finite differences
    :type h: float
    :returns: Numerical gradient approximation
    :rtype: numpy.ndarray
    
    Examples:
        >>> def f(x): return x**2
        >>> x = np.array([2.0])
        >>> grad = finite_difference_gradient(f, x)
        >>> print(grad)  # Should be close to [4.0]
    """
    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)
    
    # Compute gradient for each dimension
    for i in range(x.size):
        # Create perturbation vectors
        x_plus = x.copy()
        x_minus = x.copy()
        
        # Perturb the i-th element
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        
        # Compute finite difference
        # Handle both scalar and array outputs
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        # If function returns array, take the sum (for activation functions)
        if np.asarray(f_plus).ndim > 0:
            f_plus = np.sum(f_plus)
            f_minus = np.sum(f_minus)
        
        gradient.flat[i] = (f_plus - f_minus) / (2 * h)
    
    return gradient


def finite_difference_jacobian(func, x, h=1e-5):
    """
    Compute Jacobian matrix using finite differences.
    
    This computes the Jacobian matrix of a vector-valued function
    using finite differences. Useful for testing neural network
    gradient computations.
    
    :param func: Vector-valued function to compute Jacobian for
    :type func: callable
    :param x: Point at which to compute Jacobian
    :type x: numpy.ndarray
    :param h: Step size for finite differences
    :type h: float
    :returns: Jacobian matrix (output_size × input_size)
    :rtype: numpy.ndarray
    
    Examples:
        >>> def f(x): return np.array([x[0]**2, x[1]**3])
        >>> x = np.array([2.0, 3.0])
        >>> jacobian = finite_difference_jacobian(f, x)
        >>> print(jacobian)  # Should be close to [[4, 0], [0, 27]]
    """
    x = np.asarray(x, dtype=float)
    output = func(x)
    output = np.asarray(output, dtype=float)
    
    # Initialize Jacobian matrix
    jacobian = np.zeros((output.size, x.size))
    
    # Compute gradient for each input dimension
    for i in range(x.size):
        # Create perturbation vectors
        x_plus = x.copy()
        x_minus = x.copy()
        
        # Perturb the i-th element
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        
        # Compute finite difference
        output_plus = func(x_plus)
        output_minus = func(x_minus)
        
        jacobian[:, i] = (output_plus - output_minus).flatten() / (2 * h)
    
    return jacobian


def verify_gradient_implementation(analytical_grad, numerical_grad, rtol=1e-5, atol=1e-8):
    """
    Verify that analytical gradient matches numerical gradient.
    
    This function compares analytical and numerical gradients to ensure
    the analytical implementation is correct. This is crucial for
    debugging gradient computations in neural networks.
    
    :param analytical_grad: Analytically computed gradient
    :type analytical_grad: numpy.ndarray
    :param numerical_grad: Numerically computed gradient
    :type numerical_grad: numpy.ndarray
    :param rtol: Relative tolerance for comparison
    :type rtol: float
    :param atol: Absolute tolerance for comparison
    :type atol: float
    :returns: True if gradients match within tolerance
    :rtype: bool
    :raises ValueError: If gradient dimensions don't match
    
    Examples:
        >>> analytical = np.array([4.0, 6.0])
        >>> numerical = np.array([4.0001, 6.0001])
        >>> verify_gradient_implementation(analytical, numerical)
        True
    """
    # Convert to numpy arrays if needed
    analytical_grad = np.asarray(analytical_grad)
    numerical_grad = np.asarray(numerical_grad)
    
    # Check dimension compatibility
    if analytical_grad.shape != numerical_grad.shape:
        raise ValueError(
            f"Gradient dimension mismatch: analytical gradient shape {analytical_grad.shape} "
            f"does not match numerical gradient shape {numerical_grad.shape}"
        )
    
    try:
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def dist2(X1, X2):
    """
    Return the squared distance matrix between two 2-D arrays.
    
    Key insight: ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
    
    Why? Expand (x-y)·(x-y) = x·x - 2x·y + y·y
    """

    return (np.sum(X1*X1, axis=1, keepdims=True)
            + np.sum(X2*X2, axis=1) 
            - 2*X1@X2.T)

def generate_cluster_data(n_points_per_cluster=30):
    """Generate synthetic data with clear cluster structure for educational purposes"""
    # Define cluster centres in 2D space
    cluster_centres = np.array([[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]])
    
    # Generate data points around each center
    data_points = []
    for center in cluster_centres:
        # Generate points with some spread around each center
        cluster_points = np.random.normal(loc=center, scale=0.8, size=(n_points_per_cluster, 2))
        data_points.append(cluster_points)
    
    return np.vstack(data_points)
    
