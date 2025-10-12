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
