"""
mlai.plot
=========

Plotting utilities and visualization functions for the MLAI library.

This module provides a wide range of plotting functions for illustrating
machine learning concepts, model fits, matrix visualizations, and more.
It is designed to support both teaching and research by offering
publication-quality figures and interactive visualizations.

Key features:
- Matrix and covariance visualizations
- Regression and classification plots
- Model fit diagnostics (RMSE, holdout, cross-validation)
- Neural network diagrams
- Utility functions for figure generation

Dependencies:
- numpy
- matplotlib
- (optional) daft, IPython, mpl_toolkits.mplot3d

Some functions expect models following the MLAI interface (e.g., LM, GP).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import IPython
from mpl_toolkits.mplot3d import Axes3D


try:
    import daft
except ImportError:
    pass

import mlai as ma
from mlai.mlai import LM

# Check for GPy availability
GPY_AVAILABLE = True
try:
    import GPy
except ImportError:
    GPY_AVAILABLE = False


tau = 2*np.pi

three_figsize = (10, 3)
two_figsize = (10, 5)
one_figsize = (5, 5)
big_figsize = (7, 7)
wide_figsize = (7, 3.5)
big_wide_figsize = (10, 6)
hcolor = [1., 0., 1.] # highlighting color

notation_map={'variance': r'\\alpha',
           'lengthscale': r'\\ell',
           'period':r'\\omega'}

def pred_range(x, portion=0.2, points=200, randomize=False):
    """
    Generate a range of prediction points based on the input array x.
    :param x: Input array (1D or 2D, numeric).
    :param portion: Fraction of the span to extend beyond min/max (default: 0.2).
    :param points: Number of points in the generated range (default: 200).
    :param randomize: If True, randomly shuffle the generated points (default: False).
    :returns: Numpy array of prediction points.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Input array x must not be empty.")
    # Flatten to 1D if possible
    if x.ndim > 1:
        x = x.flatten()
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Input array x must be numeric.")
    span = np.max(x.flatten()) - np.min(x.flatten())
    xt = np.linspace(np.min(x.flatten()) - portion * span, np.max(x.flatten()) + portion * span, points)[:, np.newaxis]
    if not randomize:
        return xt
    else:
        return xt + np.random.randn(points, 1) * span / float(points)

# def write_plots(filename=filename, filebase, directory=None, width=700, height=500, kwargs):
#     """Display a series of plots controlled by sliders. The function relies on Python string format functionality to index through a series of plots."""
#     args = collections.OrderedDict(kwargs)
    
#     def write_figure(filebase, directory, **kwargs):
#         """Helper function to load in the relevant plot for display."""
#         filename = filebase.format(**kwargs)
#         if directory is not None:
#             filename = directory + '/' + filename
#         return "<img src='{filename}'>".format(filename=filename)
#     meta = '{data-transition="None"}'
#     out = '### ' + meta
#     for name, val in kwargs.items():
#         if isinstance(val,list) or isinstance(
#         out += '\\n\n' + write_figure(filebase=filebase, directory=directory, **kwargs) + '\\n\n'
#         for 
#     interact(show_figure, filebase=fixed(filebase), directory=fixed(directory), **kwargs)


def matrix(A, ax=None,
           bracket_width=3,
           bracket_style='square',
           type='values',
           colormap=None,
           highlight=False,
           highlight_row=None,
           highlight_col=None,
           highlight_width=3,
           highlight_color=[0,0,0],
           prec = '.3',
           zoom=False,
           zoom_row=None,
           zoom_col=None,
           bracket_color=[0,0,0],
           fontsize=16):
    """
    Plot a matrix with optional highlighting and custom brackets.

    :param A: Matrix to plot (2D numpy array or list of lists).
    :param ax: Matplotlib axis to draw the plot on (optional).
    :param bracket_width: Width of the bracket lines (default: 3).
    :param bracket_style: Style of brackets ('square' or 'round', default: 'square').
    :param type: Display type ('values', 'entries', etc., default: 'values').
    :param colormap: Colormap for matrix values (optional).
    :param highlight: Whether to highlight a row/column (default: False).
    :param highlight_row: Row to highlight (optional).
    :param highlight_col: Column to highlight (optional).
    :param highlight_width: Width of highlight lines (default: 3).
    :param highlight_color: Color for highlights (default: black).
    :param prec: String precision for values (default: '.3').
    :param zoom: Whether to zoom into a submatrix (default: False).
    :param zoom_row: Row index for zoom (optional).
    :param zoom_col: Column index for zoom (optional).
    :param bracket_color: Color for brackets (default: black).
    :param fontsize: Font size for text (default: 16).
    :returns: Matplotlib axis with the matrix plot.
    """
    
    if ax is None:
        ax = plt.gca()

    if colormap is not None:
        plt.set_cmap(colormap) 

    A = np.asarray(A)
    
    nrows = A.shape[0]
    ncols = A.shape[1]
    
  
    x_lim = np.array([-0.75, ncols-0.25])
    y_lim = np.array([-0.75, nrows-0.25])
  
    ax.cla()
    handle=[]
    if type == 'image':
        handle =  ax.matshow(A)
    elif type == 'imagesc':
        handle =  ax.images(A, [np.min(np.array([np.min(A.flatten()), 0])), np.max(A.flatten())])
    elif type == 'values':
        for i in range(nrows):
            for j in range(ncols):
                # Convert to float for proper formatting
                val = float(A[i, j])
                handle.append(ax.text(j, i, '{val:{prec}}'.format(val=val, prec=prec), horizontalalignment='center', fontsize=fontsize))
    elif type == 'entries':
        for i in range(nrows):
            for j in range(ncols):
                if isinstance(A[i,j], str):
                    handle.append(ax.text(j, i, A[i, j], horizontalalignment='center', fontsize=fontsize))
                    
                else:  
                    handle.append(ax.text(j, i, ' ', horizontalalignment='center', fontsize=fontsize))
    elif type == 'patch':
        for i in range(nrows):
            for j in range(ncols):
                # Convert to float for proper color handling
                color_val = float(A[i, j])
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=color_val*np.array([1, 1, 1]))))
    elif type == 'colorpatch':
        for i in range(nrows):
            for j in range(ncols):
                # Convert boolean arrays to RGB values (0 or 1)
                rgb = np.array([float(A[i, j, 0]), float(A[i, j, 1]), float(A[i, j, 2])])
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=rgb)))
                
                
    if bracket_style == 'boxes':
        x_lim = np.array([-0.5, ncols-0.5])
        ax.set_xlim(x_lim)
        y_lim = np.array([-0.5, nrows-0.5])
        ax.set_ylim(y_lim)
#        for i in range(nrows+1):
#            ax.add_line(plt.axhline(y=i-.5, #xmin=-0.5, xmax=ncols-0.5, 
#                 color=bracket_color))
#        for j in range(ncols+1):
#            ax.add_line(plt.axvline(x=j-.5, #ymin=-0.5, ymax=nrows-0.5, 
#                 color=bracket_color))
    elif bracket_style == 'square':
        tick_length = 0.25
        ax.plot([x_lim[0]+tick_length,
                     x_lim[0], x_lim[0],
                     x_lim[0]+tick_length],
                    [y_lim[0], y_lim[0],
                     y_lim[1], y_lim[1]],
                    linewidth=bracket_width,
                    color=np.array(bracket_color))
        ax.plot([x_lim[1]-tick_length, x_lim[1],
                              x_lim[1], x_lim[1]-tick_length],
                             [y_lim[0], y_lim[0], y_lim[1],
                              y_lim[1]],
                             linewidth=bracket_width, color=np.array(bracket_color))
      
    if highlight:       
        h_row = highlight_row if highlight_row is not None else ':'
        h_col = highlight_col if highlight_col is not None else ':'
        # Expand ':' to full range
        if h_row == ':':
            h_row = [0, nrows-1]
        elif isinstance(h_row, int):
            h_row = [h_row, h_row]
        elif isinstance(h_row, list):
            if len(h_row) == 1:
                h_row = [h_row[0], h_row[0]]
        if h_col == ':':
            h_col = [0, ncols-1]
        elif isinstance(h_col, int):
            h_col = [h_col, h_col]
        elif isinstance(h_col, list):
            if len(h_col) == 1:
                h_col = [h_col[0], h_col[0]]
        h_col = [int(x) for x in h_col]
        h_row = [int(x) for x in h_row]
        h_col.sort()
        h_row.sort()
        ax.add_line(plt.Line2D([h_col[0]-0.5, h_col[0]-0.5,
                              h_col[1]+0.5, h_col[1]+0.5,
                              h_col[0]-0.5],
                             [h_row[0]-0.5, h_row[1]+0.5,
                              h_row[1]+0.5, h_row[0]-0.5,
                              h_row[0]-0.5], color=highlight_color,
                               linewidth=highlight_width))
                    
    if zoom:      
        z_row = zoom_row if zoom_row is not None else ':'
        z_col = zoom_col if zoom_col is not None else ':'
        # Expand ':' to full range
        if z_row == ':':
            z_row = [0, nrows-1]
        elif isinstance(z_row, int):
            z_row = [z_row, z_row]
        elif isinstance(z_row, list):
            if len(z_row) == 1:
                z_row = [z_row[0], z_row[0]]
        if z_col == ':':
            z_col = [0, ncols-1]
        elif isinstance(z_col, int):
            z_col = [z_col, z_col]
        elif isinstance(z_col, list):
            if len(z_col) == 1:
                z_col = [z_col[0], z_col[0]]
        z_col = [int(x) for x in z_col]
        z_row = [int(x) for x in z_row]
        z_col.sort()
        z_row.sort()
        x_lim = [z_col[0]-0.5, z_col[1]+0.5]
        y_lim = [z_row[0]-0.5, z_row[1]+0.5]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis() #axis ij, axis equal, axis off

             
    return handle 


def base_plot(K, ind=[0, 1], ax=None,
              contour_color=[0., 0., 1],
              contour_style='-',
              contour_size=4,
              contour_markersize=4,
              contour_marker='x',
              fontsize=20):
    """
    Plot a base contour for a covariance matrix.

    :param K: Covariance matrix (2D numpy array).
    :param ind: Indices of variables to plot (default: [0, 1]).
    :param ax: Matplotlib axis to draw the plot on (optional).
    :param contour_color: Color for the contour (default: blue).
    :param contour_style: Line style for the contour (default: '-').
    :param contour_size: Line width for the contour (default: 4).
    :param contour_markersize: Marker size for the contour (default: 4).
    :param contour_marker: Marker style (default: 'x').
    :param fontsize: Font size for labels (default: 20).
    :returns: Matplotlib axis with the contour plot.
    """

    blackcolor = [0,0,0]
    if ax is None:
        ax = plt.gca()
    v, U = np.linalg.eig(K[ind][:, ind])
    r = np.sqrt(v)
    theta = np.linspace(0, 2*np.pi, 200)[:, np.newaxis]
    xy = np.dot(np.concatenate([r[0]*np.sin(theta), r[1]*np.cos(theta)], axis=1),U.T)
    cont = plt.Line2D(xy[:, 0], xy[:, 1],
                      linewidth=contour_size,
                      linestyle=contour_style,
                      color=contour_color)
    cent = plt.Line2D([0.], [0.],
                      marker=contour_marker,
                      color=contour_color,
                      linewidth=contour_size,
                      markersize=contour_markersize)

    ax.add_line(cont)
    ax.add_line(cent)

    thandle = []
    thandle.append(ax.set_xlabel('$f_{' + str(ind[1]+1)+ '}$',
                   fontsize=fontsize))
    thandle.append(ax.set_ylabel('$f_{' + str(ind[0]+1)+ '}$',
                   fontsize=fontsize))
    
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    x_lim = [-1.5, 1.5]
    y_lim = [-1.5, 1.5]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.add_line(plt.Line2D(x_lim, [0, 0], color=blackcolor))
    ax.add_line(plt.Line2D([0, 0], y_lim, color=blackcolor))

    ax.set_aspect('equal')
    
    return cont, thandle, cent 

def covariance_capacity(rotate_angle=np.pi/4,
                        lambda1=0.5,
                        lambda2=0.3,
                        diagrams='../diagrams/gp',
                        fill_color = [1., 1., 0.],
                        black_color = [0., 0., 0.],
                        blue_color = [0., 0., 1.],
                        magenta_color = [1., 0., 1.]):
    """
    Visualize the capacity of a covariance matrix by plotting its eigenvalues and eigenvectors.

    :param rotate_angle: Angle to rotate the covariance ellipse (default: pi/4).
    :param lambda1: First eigenvalue (default: 0.5).
    :param lambda2: Second eigenvalue (default: 0.3).
    :param diagrams: Directory to save the plot (default: '../diagrams/gp').
    :param fill_color: Fill color for the ellipse (default: yellow).
    :param black_color: Color for axes and lines (default: black).
    :param blue_color: Color for one eigenvector (default: blue).
    :param magenta_color: Color for the other eigenvector (default: magenta).
    """

    counter = 0

    fig, ax = plt.subplots(figsize=big_figsize)
    ax.set_axis_off()
    cax = fig.add_axes([0., 0., 1., 1.])
    cax.set_axis_off()

    cax.set_xlim([0., 1.])
    cax.set_ylim([0., 1.])

    # Matrix label axes
    tax2 = fig.add_axes([0, 0.47, 0.1, 0.1])
    tax2.set_xlim([0, 1.])
    tax2.set_ylim([0, 1.])
    tax2.set_axis_off()
    label_eigenvalue = tax2.text(0.5, 0.5, r'$\\Lambda=$', fontsize=20)

    ax = fig.add_axes([0.5, 0.25, 0.5, 0.5])
    ax.set_xlim([-0.25, 0.6])
    ax.set_ylim([-0.25, 0.6])
    from matplotlib.patches import Polygon
    pat_hand = ax.add_patch(Polygon(np.column_stack(([0, 0, lambda1, lambda1], 
                        [0, lambda2, lambda2, 0])), 
                        facecolor=fill_color, 
                        edgecolor=black_color, 
                        visible=False))
    data = pat_hand.get_path().vertices
    rotation_matrix = np.asarray([[np.cos(rotate_angle), -np.sin(rotate_angle)], 
                                  [np.sin(rotate_angle),  np.cos(rotate_angle)]])
    new = np.dot(rotation_matrix,data.T)
    pat_hand = ax.add_patch(Polygon(np.column_stack(([0, 0, lambda1, lambda1], 
                                                     [0, lambda2, lambda2, 0])), 
                        facecolor=fill_color, 
                        edgecolor=black_color, 
                        visible=False))
    pat_hand_rot = ax.add_patch(Polygon(new.T, 
                           facecolor=fill_color, 
                           edgecolor=black_color))
    pat_hand_rot.set(visible=False)

    # 3D box
    pat_hand3 = [ax.add_patch(Polygon(np.column_stack(([0, -0.2*lambda1, 0.8*lambda1, lambda1], 
                                          [0, -0.2*lambda2, -0.2*lambda2, 0])), 
                         facecolor=fill_color, 
                         edgecolor=black_color))]

    pat_hand3.append(ax.add_patch(Polygon(np.column_stack(([0, -0.2*lambda1, -0.2*lambda1, 0], 
                                              [0, -0.2*lambda2, 0.8*lambda2, lambda2])), 
                             facecolor=fill_color, 
                             edgecolor=black_color)))

    pat_hand3.append(ax.add_patch(Polygon(np.column_stack(([-0.2*lambda1, 0, lambda1, 0.8*lambda1], 
                                              [0.8*lambda2, lambda2, lambda2, 0.8*lambda2])), 
                             facecolor=fill_color,
                             edgecolor=black_color)))
    pat_hand3.append(ax.add_patch(Polygon(np.column_stack(([lambda1, 0.8*lambda1, 0.8*lambda1, lambda1], 
                                              [lambda2, 0.8*lambda2, -0.2*lambda2, 0])), 
                             facecolor=fill_color, 
                             edgecolor=black_color)))

    for hand in pat_hand3:
        hand.set(visible=False)

    ax.set_aspect('equal')
    ax.set_axis_off()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    ar_one = ax.arrow(x=0, y=0, dx=lambda1, dy=0, head_width=0.03)
    ar_two = ax.arrow(x=0, y=0, dx=0, dy=lambda2, head_width=0.03)
    ar_three = ax.arrow(x=0, y=0, dx=-0.2*lambda1, dy=-0.2*lambda2, head_width=0.03)
    ar_one_text = ax.text(0.5*lambda1, -0.05*yspan, 
                          '$\\lambda_1$', 
                          horizontalalignment='center',
                         fontsize=14)
    ar_two_text = ax.text(-0.05*xspan, 0.5*lambda2, 
                          '$\\lambda_2$', 
                          horizontalalignment='center',
                         fontsize=14)
    ar_three_text = ax.text(-0.05*xspan-0.1*lambda1, -0.1*lambda2+0.05*yspan, 
                            '$\\lambda_3$', 
                            horizontalalignment='center',
                           fontsize=14)

    ar_one.set(linewidth=3, 
               visible=False, 
               color=blue_color)
    ar_one_text.set(visible=False)

    ar_two.set(linewidth=3, 
               visible=False, 
               color=blue_color)
    ar_two_text.set(visible=False)

    ar_three.set(linewidth=3, 
                 visible=False, 
                 color=blue_color)
    ar_three_text.set(visible=False)


    matrix_ax = fig.add_axes([0.2, 0.35, 0.3, 0.3])
    matrix_ax.set_aspect('equal')
    matrix_ax.set_axis_off()
    eigenvals = [['$\\lambda_1$', '$0$'],['$0$', '$\\lambda_2$']]
    matrix(eigenvals, 
           matrix_ax, 
           bracket_style='square', 
           type='entries', 
           bracket_color=black_color)


    # First arrow
    matrix_ax.cla()
    matrix(eigenvals, 
           matrix_ax, 
           bracket_style='square', 
           type='entries',
           highlight=True,
           highlight_row=[0, 0],
           highlight_col=':',
           highlight_color=magenta_color,
           bracket_color=black_color)

    ar_one.set(visible=True)
    ar_one_text.set(visible=True)

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams, transparent=True)
    counter += 1

    # Second arrow
    matrix_ax.cla()
    matrix(eigenvals, 
           matrix_ax, 
           bracket_style='square', 
           type='entries', 
           highlight=True,
           highlight_row=[1,1],
           highlight_col=':',
           highlight_color=magenta_color,
           bracket_color=black_color)

    ar_two.set(visible=True)
    ar_two_text.set(visible=True)

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1

    matrix_ax.cla()
    matrix(eigenvals, matrix_ax, 
           bracket_style='square', 
           type='entries', 
           bracket_color=black_color)

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1


    tax = fig.add_axes([0.1, 0.1, 0.8, 0.1])
    tax.set_axis_off()
    tax.set_xlim([0, 1])
    tax.set_ylim([0, 1])
    det_text = tax.text(0.5, 0.5,
                    '$\\det{\\Lambda} = \\lambda_1 \\lambda_2$', 
                    horizontalalignment='center',
                       fontsize=20)
    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1

    pat_hand.set(visible=True)
    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1

    det_text_plot = ax.text(0.5*lambda1, 
                                  0.5*lambda2, 
                                  '$\\det{\\Lambda}$', 
                                  horizontalalignment='center', fontsize=20)

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1


    eigenvals2 = [['$\\lambda_1$', '$0$', '$0$'],
                  ['$0$', '$\\lambda_2$', '$0$'],
                  ['$0$', '$0$', '$\\lambda_3$']]

    matrix_ax.cla()
    matrix(eigenvals2, matrix_ax, 
           bracket_style='square', 
           type='entries',
           highlight=True,
           highlight_row=[2,2],
           highlight_col=':',
           highlight_color=magenta_color)

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1


    ar_three.set(visible=True)
    ar_three_text.set(visible=True)
    for hand in pat_hand3:
        hand.set(visible=True)
    det_text.set(text='$\\det{\\Lambda} = \\lambda_1 \\lambda_2\\lambda_3$', 
                 fontsize=20, 
                 horizontalalignment='center')

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1

    matrix_ax.cla()
    matrix(eigenvals, 
           matrix_ax, 
           bracket_style='square', 
           type='entries', 
           bracket_color=black_color)

    ar_three.set(visible=False)
    ar_three_text.set(visible=False)
    for hand in pat_hand3:
        hand.set(visible=False)
    det_text.set(text='$\\det{\\Lambda} = \\lambda_1 \\lambda_2$')

    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)
    counter += 1

    det_text.set(text='$\\det{\\mathbf{R}\\Lambda} = \\lambda_1 \\lambda_2$')
    label_eigenvalue.set(label='\\Large $\\mathbf{R}\\Lambda=$')

    import matplotlib.transforms as mtransforms

    det_text.set(text='$\\det{\\mathbf{R}\\Lambda} = \\lambda_1 \\lambda_2$')
    label_eigenvalue.set(text='$\\mathbf{R}\\Lambda=$')

    trans_data =  mtransforms.Affine2D().rotate_deg(rotate_angle*180/np.pi) + ax.transData

    ar_one.set_transform(trans_data)
    ar_one_text.set_transform(trans_data)
    ar_two.set_transform(trans_data)
    ar_two_text.set_transform(trans_data)
    det_text_plot.set_transform(trans_data)
    pat_hand_rot.set(visible=True)
    pat_hand.set(visible=False)

    pat_hand_rot.set(visible=True)
    pat_hand.set(visible=False)

    W = [['$w_{1, 1}$', '$w_{1, 2}$'],[ '$w_{2, 1}$', '$w_{2, 2}$']]
    matrix(W, 
           matrix_ax, 
           bracket_style='square', 
           type='entries', 
           bracket_color=black_color)


    file_name = 'gp-optimise-determinant{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams)

def prob_diagram(fontsize=20, diagrams='../diagrams'):
    """
    Plot a diagram demonstrating marginal and joint probabilities.

    :param fontsize: Font size to use in the plot (default: 20).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    marg = 0.05 # Distance between lines and boxes
    indent = 0.1 # indent of n indicators
    axis_indent = 0.3 # Axis indent.

    x = np.random.randn(100, 1)+4
    y = np.random.randn(100, 1)+2.5

    fig, ax =plt.subplots(figsize=big_figsize)

    # Basic plot set up.    
    a = ax.plot(x, y, 'x', color = [1, 0, 0])
    plt.axis('off')
    ax.set_xlim([0-2*marg, 6+2*marg])
    ax.set_ylim([0-2*marg, 4+2*marg])
    #ax.set_visible(False)
    for i in range(7):
        ax.plot([i, i], [0, 5], color=[0, 0, 0])
    for i in range(5):
        ax.plot([0, 7], [i, i], color=[0, 0, 0])

    for i in range(1, 5):
        ax.text(-axis_indent, i-.5, str(i), horizontalalignment='center', fontsize=fontsize)

    for i in range(1,7):
        ax.text(i-0.5, -axis_indent, str(i), horizontalalignment='center', fontsize=fontsize)

    # Box for y=4
    ax.plot([-marg, 6+marg, 6+marg, -marg, -marg], [3-marg, 3-marg, 4+marg, 4+marg, 3-marg], linestyle=':', linewidth=2, color=[1, 0, 0])
    ax.text(0.5, 4-indent, '$n_{Y=4}$', horizontalalignment='center', fontsize=fontsize)

    # Box for x=5
    ax.plot([4-marg, 5+marg, 5+marg, 4-marg, 4-marg], [-marg, -marg, 4+marg, 4+marg, -marg], linestyle='--', linewidth=2, color=[1, 0, 0])
    ax.text(4.5, 4-indent, '$n_{X=5}$', horizontalalignment='center', fontsize=fontsize)

    # Box for x=3, y=4
    ax.plot([2-2*marg, 3+2*marg, 3+2*marg, 2-2*marg, 2-2*marg], [3-2*marg, 3-2*marg, 4+2*marg, 4+2*marg, 3-2*marg], linestyle='--', linewidth=2, color=[1, 0, 1])
    ax.text(2.5, 4-indent, '$n_{X=3, Y=4}$', horizontalalignment='center', fontsize=fontsize)


    plt.text(1.5, 0.5, '$N$ crosses total', horizontalalignment='center', fontsize=fontsize)

    plt.text(3, -2*axis_indent, '$X$', fontsize=fontsize)
    plt.text(-2*axis_indent, 2, '$Y$', fontsize=fontsize)
    ma.write_figure('prob_diagram.svg', directory=diagrams, transparent=True)


def bernoulli_urn(ax, diagrams='../diagrams'):
    """
    Plot the urn of Jacob Bernoulli's analogy for the Bernoulli distribution.

    :param ax: Matplotlib axis to draw the plot on.
    :param diagrams: Directory to save the diagram (default: '../diagrams').
    """
    black_prob = 0.3
    ball_radius = 0.1

    ax.plot([0, 0, 1, 1], [1, 0, 0, 1], linewidth=3, color=[0,0,0])
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    t = np.linspace(0, 2*np.pi, 24)
    rows = 4
    cols = round(1/ball_radius)
    last_row_cols = 3
    for row in range(rows):
        if row == rows-1:
          cols = last_row_cols

        for col in range(cols):
            ball_x = col*2*ball_radius + ball_radius
            ball_y = row*2*ball_radius + ball_radius
            x = ball_x*np.ones(t.shape) + ball_radius*np.sin(t)
            y = ball_y*np.ones(t.shape) + ball_radius*np.cos(t)

            if np.random.rand()<black_prob:
                ball_color = [0, 0, 0]
            else: 
                ball_color = [1, 0, 0]
            plt.sca(ax)
            circle = plt.Circle((ball_x, ball_y), ball_radius, fill=True, color=ball_color)
            ax.add_artist(circle)

    ma.write_figure('bernoulli-urn.svg', directory=diagrams, transparent=True)

def bayes_billiard(ax, diagrams='../diagrams'):
    """
    Plot a series of figures representing Thomas Bayes' billiard table for the Bernoulli distribution representation.
    
    :param ax: Matplotlib axis to draw the plot on.
    :param diagrams: Directory to save the diagrams (default: '../diagrams').
    """
    black_prob = 0.3
    ball_radius = 0.1

    ax.plot([0, 0, 1, 1], [1, 0, 0, 1], linewidth=3, color=[0,0,0])
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ma.write_figure('bayes-billiard000.svg', directory=diagrams, transparent=True)

    ball_x = np.random.uniform(size=1)[0]
    ball_y = 0.5
    black_color = [0, 0, 0]
    red_color = [1, 0, 0]
    #r = 0.1
    #t = np.linspace(0, 2*np.pi, 24)

    #x = ball_x*np.ones(t.shape) + ball_radius*np.sin(t)
    #y = ball_y*np.ones(t.shape) + ball_radius*np.cos(t)

    circle = plt.Circle((ball_x, ball_y), ball_radius, fill=True, color=black_color)
    ax.add_artist(circle)

    ma.write_figure('bayes-billiard001.svg', directory=diagrams, transparent=True)

    ax.plot([ball_x, ball_x], [0, 1], linestyle=':', linewidth=3, color=black_color)

    ma.write_figure('bayes-billiard002.svg', directory=diagrams, transparent=True)
    counter = 2
    for ball_x in np.random.uniform(size=7):
        counter += 1
        circle = plt.Circle((ball_x, ball_y), ball_radius, fill=True, color=red_color)
        ax.add_artist(circle)
        ma.write_figure('bayes-billiard{counter:0>3}.svg'.format(counter=counter),
                          directory=diagrams,
                          transparent=True)
        circle.remove()

            
def hyperplane_coordinates(w, b, plot_limits):
    """
    Helper function for plotting the decision boundary of the perceptron.

    :param w: The weight vector for the perceptron.
    :param b: The bias parameter for the perceptron.
    :param plot_limits: Dictionary containing 'x' and 'y' plot limits.
    :returns: Tuple of (x0, x1) coordinates for the hyperplane line.
    """
    if abs(w[1])>abs(w[0]):
        # If w[1]>w[0] in absolute value, plane is likely to be leaving tops of plot.
        x0 = plot_limits['x']
        x1 = -(b + x0*w[0])/w[1]
    else:
        # otherwise plane is likely to be leaving sides of plot.
        x1 = plot_limits['y']
        x0 = -(b + x1*w[1])/w[0]
    return x0, x1

def init_perceptron(f, ax, x_plus, x_minus, w, b, fontsize=18):
    """
    Initialize a plot for showing the perceptron decision boundary.

    :param f: Matplotlib figure object.
    :param ax: Array of matplotlib axes (should have 2 axes).
    :param x_plus: Positive class data points (numpy array).
    :param x_minus: Negative class data points (numpy array).
    :param w: Weight vector for the perceptron.
    :param b: Bias parameter for the perceptron.
    :param fontsize: Font size for labels and titles (default: 18).
    :returns: Dictionary containing plot handles for updating.
    """
    h = {}

    ax[0].set_aspect('equal')
    # Plot the data again
    ax[0].plot(x_plus[:, 0], x_plus[:, 1], 'rx')
    ax[0].plot(x_minus[:, 0], x_minus[:, 1], 'go')
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)
    strt = -b/w[1]

    norm = w[0]*w[0] + w[1]*w[1]
    offset0 = -w[0]/norm*b
    offset1 = -w[1]/norm*b
    h['arrow'] = ax[0].arrow(offset0, offset1, offset0+w[0], offset1+w[1], head_width=0.2)
    # plot a line to represent the separating 'hyperplane'
    h['plane'], = ax[0].plot(x0, x1, 'b-')
    ax[0].set_xlim(plot_limits['x'])
    ax[0].set_ylim(plot_limits['y'])
    ax[0].set_xlabel('$x_0$', fontsize=fontsize)
    ax[0].set_ylabel('$x_1$', fontsize=fontsize)
    h['iter'] = ax[0].set_title('Update 0')
    
    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')
    return h

def update_perceptron(h, f, ax, x_plus, x_minus, i, w, b):
    """
    Update plots after decision boundary has changed.

    :param h: Dictionary containing plot handles from init_perceptron.
    :param f: Matplotlib figure object.
    :param ax: Array of matplotlib axes.
    :param x_plus: Positive class data points.
    :param x_minus: Negative class data points.
    :param i: Current iteration number.
    :param w: Updated weight vector.
    :param b: Updated bias parameter.
    """
    # Re-plot the hyper plane 
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)

    # Add arrow to represent hyperplane.
    h['arrow'].remove()
    del(h['arrow'])
    norm = (w[0]*w[0] + w[1]*w[1])
    offset0 = -w[0]/norm*b
    offset1 = -w[1]/norm*b
    h['arrow'] = ax[0].arrow(offset0, offset1, offset0+w[0],
                             offset1+w[1], head_width=0.2)
    
    h['plane'].set_xdata(x0)
    h['plane'].set_ydata(x1)

    h['iter'].set_text('Update ' + str(i))
    ax[1].cla()
    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')

    IPython.display.display(f)
    IPython.display.clear_output(wait=True)
    return h

def contour_error(x, y, m_center, c_center, samps=100, width=6.):
    """
    Generate error contour data for regression visualization.

    :param x: Input data points.
    :param y: Target values.
    :param m_center: Center value for slope parameter.
    :param c_center: Center value for intercept parameter.
    :param samps: Number of samples for contour generation (default: 100).
    :param width: Width of the parameter range (default: 6.0).
    :returns: Tuple of (m_vals, c_vals, E_grid) for contour plotting.
    """
    # create an array of linearly separated values around m_true
    m_vals = np.linspace(m_center-width/2., m_center+width/2., samps) 
    # create an array of linearly separated values ae
    c_vals = np.linspace(c_center-width/2., c_center+width/2., samps) 
    m_grid, c_grid = np.meshgrid(m_vals, c_vals)
    E_grid = np.zeros((samps, samps))
    for i in range(samps):
        for j in range(samps):
            E_grid[i, j] = ((y - m_grid[i, j]*x - c_grid[i, j])**2).sum()
    return m_vals, c_vals, E_grid
    
def regression_contour(f, ax, m_vals, c_vals, E_grid, fontsize=30):
    """
    Plot regression error contours.

    :param f: Matplotlib figure object.
    :param ax: Matplotlib axis object.
    :param m_vals: Slope parameter values.
    :param c_vals: Intercept parameter values.
    :param E_grid: Error values grid.
    :param fontsize: Font size for labels (default: 30).
    """
    hcont = ax.contour(m_vals, c_vals, E_grid, levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64]) # this makes the contour plot 
    plt.clabel(hcont, inline=1, fontsize=fontsize/2) # this labels the contours.

    ax.set_xlabel('$m$', fontsize=fontsize)
    ax.set_ylabel('$c$', fontsize=fontsize)

def init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star, fontsize=20):
    """
    Initialize regression visualization plots.

    :param f: Matplotlib figure object.
    :param ax: Array of matplotlib axes.
    :param x: Input data points.
    :param y: Target values.
    :param m_vals: Slope parameter values.
    :param c_vals: Intercept parameter values.
    :param E_grid: Error values grid.
    :param m_star: Optimal slope value.
    :param c_star: Optimal intercept value.
    :param fontsize: Font size for labels (default: 20).
    :returns: Dictionary containing plot handles for updating.
    """
    h = {}
    levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64]
    h['cont'] = ax[0].contour(m_vals, c_vals, E_grid, levels=levels) # this makes the contour plot on axes 0.
    plt.clabel(h['cont'], inline=1, fontsize=15)
    ax[0].set_xlabel('$m$', fontsize=fontsize)
    ax[0].set_ylabel('$c$', fontsize=fontsize)
    h['msg'] = ax[0].set_title('Error Function', fontsize=fontsize)

    # Set up plot
    h['data'], = ax[1].plot(x, y, 'r.', markersize=10)
    ax[1].set_xlabel('$x$', fontsize=fontsize)
    ax[1].set_ylabel('$y$', fontsize=fontsize)
    ax[1].set_ylim((-9, -1)) # set the y limits of the plot fixed
    ax[1].set_title('Best Fit', fontsize=fontsize)

    # Plot the current estimate of the best fit line
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plotting the current best line fit.
    y_plot = m_star*x_plot + c_star
    h['fit'], = ax[1].plot(x_plot, y_plot, 'b-', linewidth=3)
    return h

def update_regression(h, f, ax, m_star, c_star, iteration):
    """
    Update regression plots during optimization.

    :param h: Dictionary containing plot handles from init_regression.
    :param f: Matplotlib figure object.
    :param ax: Array of matplotlib axes.
    :param m_star: Current optimal slope value.
    :param c_star: Current optimal intercept value.
    :param iteration: Current iteration number.
    """
    ax[0].plot(m_star, c_star, 'g*')
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plo
    y_plot = m_star*x_plot + c_star
    
    # show the current status on the plot of the data
    h['fit'].set_ydata(y_plot)
    h['msg'].set_text('Iteration '+str(iteration))
    IPython.display.display(f)
    IPython.display.clear_output(wait=True)
    return h

def regression_contour_fit(x, y, learn_rate=0.01, m_center=1.4, c_center=-3.1, m_star = 0.0, c_star = -5.0, max_iters=1000, diagrams='../diagrams'):
    """
    Plot an evolving contour plot of regression optimisation.

    :param x: Input data points.
    :param y: Target values.
    :param learn_rate: Learning rate for optimization (default: 0.01).
    :param m_center: Center value for slope parameter (default: 1.4).
    :param c_center: Center value for intercept parameter (default: -3.1).
    :param m_star: Initial slope value (default: 0.0).
    :param c_star: Initial intercept value (default: -5.0).
    :param max_iters: Maximum number of iterations (default: 1000).
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :returns: Number of frames generated.
    """
    m_vals, c_vals, E_grid = contour_error(x, y, m_center, c_center, samps=100)

    f, ax = plt.subplots(1, 2, figsize=two_figsize) # this is to create 'side by side axes'
    # first let's plot the error surface
    handle = init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star)
    ma.write_figure('regression_contour_fit000.svg', directory=diagrams, transparent=True)

    count=0
    for i in range(max_iters): # do max_iters iterations
        # compute the gradients
        c_grad = -2*(y-m_star*x - c_star).sum()
        m_grad = -2*(x*(y-m_star*x - c_star)).sum()

        # update the parameters
        m_star = m_star - learn_rate*m_grad
        c_star = c_star - learn_rate*c_grad
        # update the location of our current best guess on the contour plot
        if i<10 or ((i<100 and not i % 10) or (i<1000 and not i % 100)): 
            handle = update_regression(handle, f, ax, m_star, c_star, i)
            count+=1
            ma.write_figure('regression_contour_fit{count:0>3}.svg'.format(count=count),
                              directory=diagrams)        
    return count

def regression_contour_sgd(x, y, learn_rate=0.01, m_center=1.4, c_center=-3.1, m_star = 0.0, c_star = -5.0, max_iters=4000, diagrams='../diagrams'):
    """
    Plot evolution of the solution of linear regression via SGD.

    :param x: Input data points.
    :param y: Target values.
    :param learn_rate: Learning rate for SGD (default: 0.01).
    :param m_center: Center value for slope parameter (default: 1.4).
    :param c_center: Center value for intercept parameter (default: -3.1).
    :param m_star: Initial slope value (default: 0.0).
    :param c_star: Initial intercept value (default: -5.0).
    :param max_iters: Maximum number of iterations (default: 4000).
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :returns: Number of frames generated.
    """
    m_vals, c_vals, E_grid = contour_error(x, y, m_center, c_center, samps=100)

    f, ax = plt.subplots(1, 2, figsize=two_figsize) # this is to create 'side by side axes'
    handle = init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star)
    count=0
    ma.write_figure('regression_sgd_contour_fit{count:0>3}.svg'.format(count=count),
                      directory=diagrams)
    for i in range(max_iters): # do max_iters iterations (parameter updates)
        # choose a random point
        index = np.random.randint(x.shape[0]-1)

        # update m
        m_star = m_star + 2*learn_rate*(x[index]*(y[index]-m_star*x[index] - c_star))
        # update c
        c_star = c_star + 2*learn_rate*(y[index]-m_star*x[index] - c_star)

        if i<10 or ((i<100 and not i % 10) or (not i % 100)): 
            handle = update_regression(handle, f, ax, m_star, c_star, i)
            count+=1
            ma.write_figure('regression_sgd_contour_fit{count:0>3}.svg'.format(count=count),
                              directory=diagrams)
    return count

def over_determined_system(diagrams='../diagrams'):
    """
    Visualize what happens in an over determined system with linear regression.

    :param diagrams: Directory to save the plots (default: '../diagrams').
    """
    x = np.array([1, 3])
    y = np.array([3, 1])

    xvals = np.linspace(0, 5, 2)

    m = (y[1]-y[0])/(x[1]-x[0])
    c = y[0]-m*x[0]

    yvals = m*xvals+c
    xvals = np.linspace(0, 5, 2)

    m = (y[1]-y[0])/(x[1]-x[0])
    c = y[0]-m*x[0]

    yvals = m*xvals+c

    ylim = np.array([0, 5])
    xlim = np.array([0, 5])

    f, ax = plt.subplots(1,1,figsize=one_figsize)
    a = ax.plot(xvals, yvals, '-', linewidth=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.xlabel('$x$', fontsize=30)
    plt.ylabel('$y$',fontsize=30)
    plt.text(4, 4, '$y=mx+c$',  horizontalalignment='center', verticalalignment='bottom', fontsize=30)
    ma.write_figure('over_determined_system001.svg', directory=diagrams, transparent=True)
    ctext = ax.text(0.15, c+0.15, '$c$',  horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    xl = np.array([1.5, 2.5])
    yl = xl*m + c
    mhand = ax.plot([xl[0], xl[1]], [np.min(yl), np.min(yl)], color=[0, 0, 0])
    mhand2 = ax.plot([np.min(xl), np.min(xl)], [yl[0], yl[1]], color=[0, 0, 0])
    mtext = ax.text(xl.mean(), np.min(yl)-0.2, '$m$',  horizontalalignment='center', verticalalignment='bottom',fontsize=20)
    ma.write_figure('over_determined_system002.svg', directory=diagrams, transparent=True)

    a2 = ax.plot(x, y, '.', markersize=20, linewidth=3, color=[1, 0, 0])
    ma.write_figure('over_determined_system003.svg', directory=diagrams, transparent=True)

    xs = 2
    ys = m*xs + c + 0.3

    ast = ax.plot(xs, ys, '.', markersize=20, linewidth=3, color=[0, 1, 0])
    ma.write_figure('over_determined_system004.svg', directory=diagrams, transparent=True)


    m = (y[1]-ys)/(x[1]-xs)
    c = ys-m*xs
    yvals = m*xvals+c

    for i in a:
        i.set_visible(False)
    for i in mhand:
        i.set_visible(False)
    for i in mhand2:
        i.set_visible(False)
    mtext.set_visible(False)
    ctext.set_visible(False)
    a3 = ax.plot(xvals, yvals, '-', linewidth=2, color=[0, 0, 1])
    for i in ast:
        i.set_color([1, 0, 0])
    ma.write_figure('over_determined_system005.svg', directory=diagrams, transparent=True)

    m = (ys-y[0])/(xs-x[0])
    c = y[0]-m*x[0]
    yvals = m*xvals+c

    for i in a3:
        i.set_visible(False)
    a4 = ax.plot(xvals, yvals, '-', linewidth=2, color=[0, 0, 1])
    for i in ast:
        i.set_color([1, 0, 0])
    ma.write_figure('over_determined_system006.svg', directory=diagrams, transparent=True)
    for i in a:
        i.set_visible(True)
    for i in a3:
        i.set_visible(True)
    ma.write_figure('over_determined_system007.svg', directory=diagrams, transparent=True)

def gaussian_of_height(diagrams='../diagrams'):
    """
    Plot a Gaussian density representing heights.

    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    "Gaussian density representing heights."
    h = np.linspace(0, 2.5, 1000)
    sigma2 = 0.0225
    mu = 1.7
    p = 1./np.sqrt(2*np.pi*sigma2)*np.exp(-(h-mu)**2/(2*sigma2**2))
    f2, ax2 = plt.subplots(figsize=wide_figsize)
    ax2.plot(h, p, 'b-', linewidth=3)
    ylim = (0, 3)
    ax2.vlines(mu, ylim[0], ylim[1], colors='r', linewidth=3)
    ax2.set_ylim(ylim)
    ax2.set_xlim(1.4, 2.0)
    ax2.set_xlabel('$h/m$', fontsize=20)
    ax2.set_ylabel('$p(h|\\mu, \\sigma^2)$', fontsize = 20)
    ma.write_figure(figure=f2, filename='gaussian_of_height.svg', directory=diagrams, transparent=True)

def marathon_fit(model, value, param_name, param_range,
                 xlim, fig, ax, x_val=None, y_val=None, objective=None,
                 diagrams='../diagrams', fontsize=20, objective_ylim=None,
                 prefix='olympic', title=None, png_plot=False, samps=130):
    """
    Plot fit of the olympic marathon data alongside error.

    :param model: Model object with a predict method and data attributes.
    :param value: Value to fit.
    :param param_name: Name of the parameter being varied.
    :param param_range: Range of parameter values.
    :param xlim: Limits for the x-axis.
    :param fig: Matplotlib figure object.
    :param ax: Array of matplotlib axes.
    :param x_val: Optional x value for highlighting (default: None).
    :param y_val: Optional y value for highlighting (default: None).
    :param objective: Objective function (optional).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    :param fontsize: Font size for labels (default: 20).
    :param objective_ylim: Y-axis limits for the objective plot (optional).
    :param prefix: Prefix for saved plot filenames (default: 'olympic').
    :param title: Title for the plot (optional).
    :param png_plot: Whether to save as PNG (default: False).
    :param samps: Number of samples for prediction (default: 130).
    """
    if title is None:
        title = model.objective_name
        
    ax[0].cla()
    ax[0].plot(model.X, model.y, 'o', color=[1, 0, 0], markersize=6, linewidth=3)
    if x_val is not None and y_val is not None:
        ax[0].plot(x_val, y_val, 'o', color=[0, 1, 0], markersize=6, linewidth=3)
        
    ylim = ax[0].get_ylim()

    x_pred = np.linspace(xlim[0], xlim[1], samps)[:, np.newaxis]
    y_pred, y_var = model.predict(x_pred)
    
    if y_var is None:
        if GPY_AVAILABLE:
            import mlai.gp_tutorial as gpt
            gpt.meanplot(x_pred, y_pred, ax=ax[0])
        else:
            ax[0].plot(x_pred, y_pred, 'b-', linewidth=2)
    else:
        y_err = np.sqrt(y_var)*2
        if GPY_AVAILABLE:
            import mlai.gp_tutorial as gpt
            gpt.gpplot(x_pred, y_pred, y_pred - y_err, y_pred + y_err, ax=ax[0])
        else:
            ax[0].plot(x_pred, y_pred, 'b-', linewidth=2)
            ax[0].fill_between(x_pred.flatten(), (y_pred - y_err).flatten(), (y_pred + y_err).flatten(), alpha=0.3)
        
    #ax[0].set_xlabel('year', fontsize=fontsize)
    ax[0].set_ylim(ylim)
    plt.sca(ax[0])

    xlim = ax[0].get_xlim()

    if objective is not None:
        #ax[1].cla()
        #params = range(*param_range)
        #for name, vals in objective.items():
        #    ax[1].plot(np.array(params), vals, 'o',
        #               color=[1, 0, 0], markersize=6, linewidth=3)
        ax[1].plot(value, objective, 'o',
                   color=[1, 0, 0], markersize=6, linewidth=3)
        if len(param_range)>2:
            xlow = param_range[0]-param_range[2]
            xhigh = param_range[1]
        else:
            xlow = param_range[0]-1
            xhigh = param_range[1]
        ax[1].set_xlim((xlow, xhigh))
        ax[1].set_ylim(objective_ylim)
        ax[1].set_xlabel(param_name.replace('_', ' '), fontsize=fontsize)
        if title is not None:
            ax[1].set_title(title, fontsize=fontsize)

    filename = '{prefix}_{name}_{param_name}{value:0>3}'.format(prefix=prefix, name=model.name, param_name=param_name, value=value)
    ma.write_figure(filename + '.svg',
                      directory=diagrams,
                      transparent=True)
    if png_plot:
        ma.write_figure(filename + '.png',
                          directory=diagrams,
                          transparent=True)



def rmse_fit(x, y, param_name, param_range,
             model=LM, #plot_objectives={'RMSE':ma.MapModel.rmse},
             objective_ylim=None, xlim=None,
             plot_fit=marathon_fit, diagrams='../diagrams', **kwargs):
    """
    Fit a model and show RMSE error.

    :param x: The input x data.
    :param y: The input y data.
    :param param_name: The parameter name to vary.
    :param param_range: The range over which to vary the parameter.
    :param model: The model to fit (default is LM).
    :param objective_ylim: The y limits for the plot of the objective.
    :param xlim: The x limits for the plot.
    :param plot_fit: Function to use for plotting the fit.
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :param **kwargs: Additional keyword arguments passed to plot_fit.
    """
    f, ax = plt.subplots(1, 2, figsize=two_figsize)
    num_data = x.shape[0]
    
    params = range(*param_range)

    count = 0
    obj = {}
    for param in params:
        m = model(x, y, **kwargs)
        m.set_param(param_name, param)
        m.fit()
        # compute appropriate objective. 
        #for name, plot_objective in plot_objectives.items():
        obj=m.rmse()#[name][count] = plot_objective(m)
            
        plot_fit(model=m, value=param, xlim=xlim,
                 param_name=param_name, param_range=param_range,
                 objective=obj, objective_ylim=objective_ylim,
                 fig=f, ax=ax, diagrams=diagrams)
        count += 1


def holdout_fit(x, y, param_name, param_range, model=LM, val_start=20,
                objective_ylim=None, xlim=None, plot_fit=marathon_fit,
                permute=True, prefix='olympic_val', diagrams='../diagrams', **kwargs):
    """
    Fit a model and show holdout error.

    :param x: The input x data.
    :param y: The input y data.
    :param param_name: The parameter name to vary.
    :param param_range: The range over which to vary the parameter.
    :param model: The model to fit (default is LM).
    :param val_start: Starting index for validation set (default: 20).
    :param objective_ylim: The y limits for the plot of the objective.
    :param xlim: The x limits for the plot.
    :param plot_fit: Function to use for plotting the fit.
    :param permute: Whether to permute the data (default: True).
    :param prefix: Prefix for saved plot filenames (default: 'olympic_val').
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :param **kwargs: Additional keyword arguments passed to plot_fit.
    """
    f, ax = plt.subplots(1, 2, figsize=two_figsize)
    num_data = x.shape[0]

    if permute:
        perm = np.random.permutation(num_data)
        x_tr = x[perm[:val_start], :]
        x_val = x[perm[val_start:], :]
        y_tr = y[perm[:val_start], :]
        y_val = y[perm[val_start:], :]
    else:
        x_tr = x[:val_start, :]
        x_val = x[val_start:, :]
        y_tr = y[:val_start, :]
        y_val = y[val_start:, :]
    num_val_data = x_val.shape[0]

    params = range(*param_range)
    ll = np.array([np.nan]*len(params))
    ss = np.array([np.nan]*len(params))
    ss_val = np.array([np.nan]*len(params))
    count = 0
    for param in params:    
        m = model(x_tr, y_tr, **kwargs)
        m.set_param(param_name, param)
        m.fit()
        f_val, _ = m.predict(x_val)
        ss[count] = m.objective()
        ss_val[count] = ((y_val-f_val)**2).mean() 
        ll[count] = m.log_likelihood()
        plot_fit(model=m, value=param, xlim=xlim,
                 param_name=param_name, param_range=param_range,
                 objective=np.sqrt(ss_val[count]), objective_ylim=objective_ylim,
                 fig=f, ax=ax, prefix=prefix,
                 title="Hold Out Validation",
                 x_val=x_val, y_val=y_val, diagrams=diagrams)
        count+=1

def loo_fit(x, y, param_name, param_range,
            model=LM, objective_ylim=None, 
            xlim=None, plot_fit=marathon_fit,
            prefix='olympic_loo', diagrams='../diagrams',
            **kwargs):
    """
    Fit a model and show leave one out error.

    :param x: The input x data.
    :param y: The input y data.
    :param param_name: The parameter name to vary.
    :param param_range: The range over which to vary the parameter.
    :param model: The model to fit (default is LM).
    :param objective_ylim: The y limits for the plot of the objective.
    :param xlim: The x limits for the plot.
    :param plot_fit: Function to use for plotting the fit.
    :param prefix: Prefix for saved plot filenames (default: 'olympic_loo').
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :param **kwargs: Additional keyword arguments passed to plot_fit.
    """
    f, ax = plt.subplots(1, 2, figsize=two_figsize)


    num_data = x.shape[0]
    num_parts = num_data
    partitions = []
    for part in range(num_parts):
        train_ind = list(range(part))
        train_ind.extend(range(part+1,num_data))
        val_ind = [part]
        partitions.append((train_ind, val_ind))

        params = range(*param_range)        
        ll = np.array([np.nan]*len(params))
        ss = np.array([np.nan]*len(params))
        ss_val = np.array([np.nan]*len(params))
        count = 0
        for param in params:
            ss_temp = 0.
            ll_temp = 0.
            ss_val_temp = 0.
            for part, (train_ind, val_ind) in enumerate(partitions):
                x_tr = x[train_ind, :]
                x_val = x[val_ind, :]
                y_tr = y[train_ind, :]
                y_val = y[val_ind, :]
                num_val_data = x_val.shape[0]
                m = model(x_tr, y_tr, **kwargs)
                m.set_param(param_name, param)
                m.fit()
                ss_temp = m.objective()
                ll_temp = m.log_likelihood()
                f_val, _ = m.predict(x_val)
                ss_val_temp += ((y_val-f_val)**2).mean() 
                plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                         objective=np.nan, objective_ylim=objective_ylim,
                         fig=f, ax=ax, prefix='olympic_loo{part:0>3}'.format(part=part),
                         x_val=x_val, y_val=y_val, diagrams=diagrams)
            ss[count] = ss_temp/(num_parts)
            ll[count] = ll_temp/(num_parts)
            ss_val[count] = ss_val_temp/(num_parts)
            plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                     objective=np.sqrt(ss_val[count]), objective_ylim=objective_ylim,
                     fig=f, ax=ax, prefix='olympic_loo{part:0>3}'.format(part=len(partitions)),
                     title="Leave One Out Validation",
                     x_val=x_val, y_val=y_val, diagrams=diagrams)
            count+=1


def cv_fit(x, y, param_name, param_range, model=LM, objective_ylim=None, 
               xlim=None, plot_fit=marathon_fit, num_parts=5, diagrams='../diagrams', **kwargs):
    """
    Fit a model and show cross validation error.

    :param x: The input x data.
    :param y: The input y data.
    :param param_name: The parameter name to vary.
    :param param_range: The range over which to vary the parameter.
    :param model: The model to fit (default is LM).
    :param objective_ylim: The y limits for the plot of the objective.
    :param xlim: The x limits for the plot.
    :param plot_fit: Function to use for plotting the fit.
    :param num_parts: Number of parts for cross-validation (default: 5).
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :param **kwargs: Additional keyword arguments passed to plot_fit.
    """
    f, ax = plt.subplots(1, 2, figsize=two_figsize)
    num_data = x.shape[0]
    partitions = []
    ind = list(np.random.permutation(num_data))
    start = 0
    for part in range(num_parts):
        end = round((float(num_data)/num_parts)*(part+1))
        train_ind = ind[:start]
        train_ind.extend(ind[end:])
        val_ind = ind[start:end]
        partitions.append((train_ind, val_ind))
        start = end

    params = range(*param_range)
    for param in params:
        ss_val_temp = 0.
        ll_temp = 0.
        ss_temp = 0.
        for part, (train_ind, val_ind) in enumerate(partitions):
            x_tr = x[train_ind, :]
            x_val = x[val_ind, :]
            y_tr = y[train_ind, :]
            y_val = y[val_ind, :]
            num_val_data = x_val.shape[0]
            m = model(x_tr, y_tr, **kwargs)
            m.set_param(param_name, param)
            m.fit()
            ss_temp += m.objective()
            ll_temp += m.log_likelihood()
            f_val, _ = m.predict(x_val)
            ss_val_temp += ((y_val-f_val)**2).mean() 
            plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                     objective=np.nan, objective_ylim=objective_ylim,
                     fig=f, ax=ax, prefix='olympic_{num_parts}cv{part:0>2}'.format(num_parts=num_parts, part=part),
                     title='{num_parts}-fold Cross Validation'.format(num_parts=num_parts),
                     x_val=x_val, y_val=y_val, diagrams=diagrams)
        ss_val = ss_val_temp/(num_parts)
        ss = ss_temp/(num_parts)
        ll = ll_temp/(num_parts)
        plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                 objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                 fig=f, ax=ax,
                 prefix='olympic_{num_parts}cv{num_partitions:0>2}'.format(num_parts=num_parts, num_partitions=num_parts),
                 title='{num_parts}-fold Cross Validation'.format(num_parts=num_parts),
                 x_val=x_val, y_val=y_val, diagrams=diagrams)
            
#################### Session 6 ####################    

def under_determined_system(diagrams='../diagrams'):
    """
    Visualize what happens in an under determined system with linear regression.

    :param diagrams: Directory to save the plots (default: '../diagrams').
    """
    x = 1.
    y = 3.
    fig, ax = plt.subplots(figsize=one_figsize)
    ax.plot(x, y, 'o', markersize=10, linewidth=3, color=[1., 0., 0.])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ylim = [0, 5]
    xlim = [0, 3]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    ma.write_figure(figure=fig, filename='under_determined_system000.svg', directory=diagrams, transparent=True, frameon=True)

    xvals = np.linspace(0, 3, 2)[:, np.newaxis]
    count=0
    for i in range(100):
        c = np.random.normal(size=(1,1))*2
        m = (y - c)/x
        yvals = m*xvals+c
        ax.plot(xvals, yvals, '-', linewidth=2, color=[0., 0., 1.])
        if i < 9 or i == 100:
            count += 1
            ma.write_figure(figure=fig, filename='under_determined_system{count:0>3}.svg'.format(count=count),
                              directory=diagrams,
                              transparent=True, frameon=True)


def bayes_update(diagrams='../diagrams'):
    """
    Visualize Bayesian updating with a simple example.

    :param diagrams: Directory to save the plots (default: '../diagrams').
    """
    fig, ax = plt.subplots(figsize=two_figsize)
    num_points = 1000
    x_max = 6
    x_min = -1

    y = np.array([[1.]])
    prior_mean = np.array([[0.]])
    prior_var = np.array([[.1]])

    noise = ma.Gaussian(offset=np.array([0.6]), scale=np.array(np.sqrt(0.05)))


    f = np.linspace(x_min, x_max, num_points)[:, np.newaxis]
    ln_prior_curve = -0.5*(np.log(2*np.pi*prior_var) + (f-prior_mean)*(f-prior_mean)/prior_var)
    ln_likelihood_curve = np.zeros(ln_prior_curve.shape)
    for i in range(num_points):
        ln_likelihood_curve[i] = noise.log_likelihood(f[i][np.newaxis, :], 
                                                      np.array([[np.finfo(float).eps]]), 
                                                      y)
    ln_marginal_likelihood = noise.log_likelihood(prior_mean, prior_var, y)

    prior_curve = np.exp(ln_prior_curve) 
    likelihood_curve = np.exp(ln_likelihood_curve)
    marginal_curve = np.exp(ln_marginal_likelihood)

    ln_posterior_curve = ln_likelihood_curve + ln_prior_curve - ln_marginal_likelihood
    posterior_curve = np.exp(ln_posterior_curve)

    g, dlnZ_dvs = noise.grad_vals(prior_mean, prior_var, y)

    nu = g*g - 2*dlnZ_dvs

    approx_var = prior_var - prior_var*prior_var*nu
    approx_mean = prior_mean + prior_var*g

    ln_approx_curve = -0.5*np.log(2*np.pi*approx_var)-0.5*(f-approx_mean)*(f-approx_mean)/approx_var

    approx_curve = np.exp(ln_approx_curve)
    noise
    xlim = [x_min, x_max] 
    ylim = [0, np.max(np.vstack([approx_curve, likelihood_curve, 
                          posterior_curve, prior_curve]).flatten())*1.1]

    fig, ax = plt.subplots(figsize=two_figsize)

    ax.set_xlim(xlim)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(ylim)

    ax.vlines(xlim[0], ylim[0], ylim[1], color=[0., 0., 0.]) 
    ax.hlines(ylim[0], xlim[0], xlim[1], color=[0., 0., 0.]) 

    ax.plot(f, prior_curve, color=[1, 0., 0.], linewidth=3)
    ax.text(3.5, 2, r'$p(c) = \\mathcal{N}(c|0, \\alpha_1)$', horizontalalignment='center', fontsize=20) 
    ma.write_figure('dem_gaussian001.svg', directory=diagrams, transparent=True)

    ax.plot(f, likelihood_curve, color=[0, 0, 1], linewidth=3)
    ax.text(3.5, 1.5,r'$p(y|m, c, x, \\sigma^2)=\\mathcal{N}(y|mx+c,\\sigma^2)$', horizontalalignment='center', fontsize=20) 
    ma.write_figure('dem_gaussian002.svg', directory=diagrams, transparent=True)

    ax.plot(f, posterior_curve, color=[1, 0, 1], linewidth=3)
    ax.text(3.5, 1, r'$p(c|y, m, x, \\sigma^2)=$', horizontalalignment='center', fontsize=20) 
    plt.text(3.5, 0.65, r'$\\mathcal{N}\\left(c|\\frac{y-mx}{1+\\sigma^2\\alpha_1},(\\sigma^{-2}+\\alpha_1^{-1})^{-1}\\right)$', horizontalalignment='center', fontsize=20)
    ma.write_figure('dem_gaussian003.svg', directory=diagrams, transparent=True)

def height_weight(h=None, w=None, muh=1.7, varh=0.0225,
                  muw=75, varw=36, diagrams='../diagrams'):
    """
    Plot height and weight data with Gaussian distributions.

    :param h: Height data (optional).
    :param w: Weight data (optional).
    :param muh: Mean height (default: 1.7).
    :param varh: Variance of height (default: 0.0225).
    :param muw: Mean weight (default: 75).
    :param varw: Variance of weight (default: 36).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, np.newaxis]
    if w is None:
        w = np.linspace(55, 95, 100)[:, np.newaxis]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)

    fig, ax = plt.subplots(1, 2, figsize=two_figsize)

    height(ax[0], h, ph)

    weight(ax[1], w, pw)
    ma.write_figure('height_weight_gaussian.svg', directory=diagrams, transparent=True)

def independent_height_weight(h=None, w=None, muh=1.7, varh=0.0225,
                              muw=75, varw=36, num_samps=20,
                              diagrams='../diagrams'):
    """
    Plot independent height and weight samples.

    :param h: Height data (optional).
    :param w: Weight data (optional).
    :param muh: Mean height (default: 1.7).
    :param varh: Variance of height (default: 0.0225).
    :param muw: Mean weight (default: 75).
    :param varw: Variance of weight (default: 36).
    :param num_samps: Number of samples to generate (default: 20).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, np.newaxis]
    if w is None:
        w = np.linspace(55, 95, 100)[:, np.newaxis]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)
    
    fig, axs = plt.subplots(2, 4, figsize=two_figsize)
    for a in axs.flatten():
        a.set_axis_off()
    ax=[]
    ax.append(plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2))
    ax.append(plt.subplot2grid((2,4), (0,3)))
    ax.append(plt.subplot2grid((2,4), (1,3)))

    ax[0].plot(muh, muw, 'x', color=[1., 0., 1.], markersize=5., linewidth=3)
    theta = np.linspace(0, tau, 100)
    xel = np.sin(theta)*np.sqrt(varh) + muh
    yel = np.cos(theta)*np.sqrt(varw) + muw
    ax[0].plot(xel, yel, '-', color=[1., 0., 1.], linewidth=3)
    ax[0].set_xlim([np.min(h), np.max(h)])
    ax[0].set_ylim([np.min(w)+10, np.max(w)-10])
    ax[0].set_yticks([65, 75, 85])
    ax[0].set_xticks([1.25, 1.7, 2.15])
    ax[0].set_xlabel('$h/m$', fontsize=20)
    ax[0].set_ylabel('$w/kg$', fontsize=20)

    ylim = ax[0].get_ylim()
    xlim = ax[0].get_xlim()
    ax[0].vlines(xlim[0], ylim[0], ylim[1], color=[0.,0.,0.])
    ax[0].hlines(ylim[0], xlim[0], xlim[1], color=[0., 0., 0.])

    height(ax[1], h, ph)
    weight(ax[2], w, pw)
    count = 0


    for i in range(num_samps):
        hval = np.random.normal(size=(1,1))*np.sqrt(varh) + muh
        wval = np.random.normal(size=(1,1))*np.sqrt(varw) + muw
        a1 = ax[1].plot(hval, 0.1, marker='o', linewidth=3, color=[1., 0., 0.])
        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1
        a2 = ax[2].plot(wval, 0.002, marker='o', linewidth=3, color=[1., 0., 0.])
        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1
        a0 = ax[0].plot(hval, wval, marker='o', linewidth=3, color=[1., 0., 0.])
        ma.write_figure(figure=fig, filename='independent_height_weight{count:0>3}.svg'.format(count=count), directory=diagrams, transparent=True)
        count+=1

        a0[0].set(color=[0.,0.,0.])
        a1[0].set(color=[0.,0.,0.])
        a2[0].set(color=[0.,0.,0.])
        
        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1

def correlated_height_weight(h=None, w=None, muh=1.7, varh=0.0225,
                             muw=75, varw=36, num_samps=20, diagrams='../diagrams'):
    """
    Plot correlated height and weight samples.

    :param h: Height data (optional).
    :param w: Weight data (optional).
    :param muh: Mean height (default: 1.7).
    :param varh: Variance of height (default: 0.0225).
    :param muw: Mean weight (default: 75).
    :param varw: Variance of weight (default: 36).
    :param num_samps: Number of samples to generate (default: 20).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, np.newaxis]
    if w is None:
        w = np.linspace(55, 95, 100)[:, np.newaxis]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)

    fig, axs = plt.subplots(2, 4, figsize=two_figsize)
    for a in axs.flatten():
        a.set_axis_off()
    ax=[]
    ax.append(plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2))
    ax.append(plt.subplot2grid((2,4), (0,3)))
    ax.append(plt.subplot2grid((2,4), (1,3)))

    covMat = np.asarray([[1, 0.995], [0.995, 1]])
    fact = np.asarray([[np.sqrt(varh), 0], [0, np.sqrt(varw)]])
    covMat = np.dot(np.dot(fact,covMat), fact)
    _, R = np.linalg.eig(covMat)

    ax[0].plot(muh, muw, 'x', color=[1., 0., 1.], markersize=5, linewidth=3)
    theta = np.linspace(0, tau, 100)
    xel = np.sin(theta)*np.sqrt(varh)
    yel = np.cos(theta)*np.sqrt(varw)
    vals = np.dot(R,np.vstack([xel, yel]))
    ax[0].plot(vals[0, :]+muh, vals[1, :]+muw, '-', color=[1., 0., 1.], linewidth=3)
    ax[0].set_xlim([np.min(h), np.max(h)])
    ax[0].set_ylim([np.min(w)+10, np.max(w)-10])
    ax[0].set_yticks([65, 75, 85])
    ax[0].set_xticks([1.25, 1.7, 2.15])
    ax[0].set_xlabel('$h/m$', fontsize=20)
    ax[0].set_ylabel('$w/kg$', fontsize=20)

    height(ax[1], h, ph)
    weight(ax[2], w, pw)
    count = 0
    for i in range(num_samps):
        vec_s = np.dot(np.dot(R,fact),np.random.normal(size=(2,1)))
        hval = vec_s[0] + muh
        wval = vec_s[1] + muw
        a1 = ax[1].plot(hval, 0.1, marker='o', linewidth=3, color=[1., 0., 0.])
        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        a2 = ax[2].plot(wval, 0.002, marker='o', linewidth=3, color=[1., 0., 0.])
        #count+=1
        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)

        a0 = ax[0].plot(hval, wval, marker='o', linewidth=3, color=[1., 0., 0.])
        #count+=1
        ma.write_figure(figure=fig, filename='correlated_height_weight{count:0>3}.svg'.format(count=count), directory=diagrams, transparent=True)
        #count+=1

        a0[0].set(color=[0.,0.,0.])
        a1[0].set(color=[0.,0.,0.])
        a2[0].set(color=[0.,0.,0.])

        #ma.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        count+=1




#################### Session 11 ####################


def two_point_pred(K, f, x, ax=None, ind=[0, 1],
                   conditional_linestyle = '-',
                   conditional_linecolor = [1., 0., 0.],
                   conditional_size = 4,
                   fixed_linestyle = '-',
                   fixed_linecolor = [0., 1., 0.],
                   fixed_size = 4, stub=None, start=0,
                   diagrams='../diagrams'):
    """
    Plot two-point prediction for Gaussian processes.

    :param K: Covariance matrix.
    :param f: Function values.
    :param x: Input points.
    :param ax: Matplotlib axis (optional).
    :param ind: Indices to plot (default: [0, 1]).
    :param conditional_linestyle: Line style for conditional (default: '-').
    :param conditional_linecolor: Color for conditional (default: red).
    :param conditional_size: Line width for conditional (default: 4).
    :param fixed_linestyle: Line style for fixed (default: '-').
    :param fixed_linecolor: Color for fixed (default: green).
    :param fixed_size: Line width for fixed (default: 4).
    :param stub: Stub parameter (optional).
    :param start: Starting index (default: 0).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)
    
    subK = K[ind][:, ind]
    f = f[ind]
    x = x[ind]

    if ax is None:
        ax = plt.gca()

    cont, t, cent = base_plot(K, ind, ax=ax)
    if stub is not None:
        ma.write_figure('{stub}{start:0>3}.svg'.format(stub=stub, start=start), directory=diagrams, transparent=True)

    x_lim = ax.get_xlim()
    cont2 = plt.Line2D([x_lim[0], x_lim[1]], [f[0], f[0]], linewidth=fixed_size, linestyle=fixed_linestyle, color=fixed_linecolor)
    ax.add_line(cont2)

    if stub is not None:
        ma.write_figure('{stub}{start:0>3}.svg'.format(stub=stub, start=start+1), directory=diagrams, transparent=True)

    # # Compute conditional mean and variance
    f2_mean = subK[0, 1]/subK[0, 0]*f[0]
    f2_var = subK[1, 1] - subK[0, 1]/subK[0, 0]*subK[0, 1]
    x_val = np.linspace(x_lim[0], x_lim[1], 200)
    pdf_val = 1/np.sqrt(2*np.pi*f2_var)*np.exp(-0.5*(x_val-f2_mean)*(x_val-f2_mean)/f2_var)
    pdf = plt.Line2D(x_val, pdf_val+f[0], linewidth=conditional_size, linestyle=conditional_linestyle, color=conditional_linecolor)
    ax.add_line(pdf)
    if stub is not None:
        ma.write_figure('{stub}{start:0>3}.svg'.format(stub=stub, start=start+2), directory=diagrams, transparent=True)
    
    obs = plt.Line2D([f[1]], [f[0]], linewidth=10, markersize=10, color=fixed_linecolor, marker='o')
    ax.add_line(obs)
    if stub is not None:
        ma.write_figure('{stub}{start:0>3}.svg'.format(stub=stub, start=start+3), directory=diagrams, transparent=True)
    

def output_augment_x(x, num_outputs):
    """
    Augment input x with output dimensions.

    :param x: Input data.
    :param num_outputs: Number of outputs.
    :returns: Augmented input data.
    """
    num_data = x.shape[0]
    x = np.tile(x, (num_outputs, 1))
    index = np.asarray([])
    for i in range(num_outputs):
        index=np.append(index, np.ones(num_data)*i)
    index = index[:, np.newaxis]
    return np.hstack((index, x))

def basis(function, x_min, x_max, fig, ax, loc, text, diagrams='./diagrams', fontsize=20, num_basis=3, num_plots=3):
    """
    Plot basis functions.

    :param function: Basis function to plot.
    :param x_min: Minimum x value.
    :param x_max: Maximum x value.
    :param fig: Matplotlib figure.
    :param ax: Matplotlib axis.
    :param loc: Location for text.
    :param text: Text to display.
    :param diagrams: Directory to save the plot (default: './diagrams').
    :param fontsize: Font size (default: 20).
    :param num_basis: Number of basis functions (default: 3).
    :param num_plots: Number of plots (default: 3).
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)
    x = np.linspace(x_min, x_max, 100)[:, None]

    basis = ma.Basis(function, num_basis)
    Phi = basis.Phi(x)
    diag=1/basis.number*(Phi*Phi).sum(1)

    colors = []
    colors.append([1, 0, 0])
    colors.append([1, 0, 1])
    colors.append([0, 0, 1])
    colors.append([0, 1, 0])
    colors.append([0, 1, 1])
    colors.append([1, 0, 1])

    # Set ylim according to max standard deviation of basis
    ylim = 2*np.asarray([-1, 1])*np.sqrt(diag.max())    
    plt.sca(ax)
    ax.set_xlim((x_min, x_max))    
    ax.set_ylim(ylim)

    ax.set_xlabel('$x$', fontsize=fontsize)
    ax.set_ylabel('$\\phi(x)$', fontsize=fontsize)
    for i in range(basis.number):
        ax.plot(x, Phi[:, i], '-', color=colors[i], linewidth=3)
        ax.text(loc[i][0], loc[i][1], text[i], horizontalalignment='center', fontsize=fontsize, color=colors[i])
        ma.write_figure(basis.function.__name__ + '_basis{num:0>3}.svg'.format(num=i), directory=diagrams, transparent=True)

    # Set ylim according to max standard deviation of basis
    plt.sca(ax)
    ax.cla()
    ylim = 3*np.asarray([-1, 1])*np.sqrt(diag.max())
    ax.set_xlabel('$x$', fontsize=fontsize) 
    ax.set_ylabel('$f(x)$', fontsize=fontsize)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim(ylim)

    f = np.dot(Phi, np.zeros((basis.number, 1)))
    a, = ax.plot(x, f, color=[0, 0, 0], linewidth=3)

    for i in range(basis.number):
        ax.plot(x.flatten(), Phi[:, i], color=colors[i], linewidth=1) 

    t = []
    for i in range(basis.number):
        t.append(ax.text(loc[i][0], loc[i][1], '$w_' + str(i) + ' = 0$',
                         horizontalalignment='center', fontsize=fontsize,
                         verticalalignment='center', color=colors[i]))

    for j in range(num_plots):
        # Sample a function
        w = np.random.normal(size=(basis.number, 1))/basis.number
        f = np.dot(Phi,w)
        a.set_ydata(f)
        for i in range(basis.number):
            t[i].set_text('$w_{ind} = {w:3.3}$'.format(ind=i, w=w[i,0]))

        ma.write_figure(basis.function.__name__ + '_function{plot_num:0>3}.svg'.format(plot_num=j), directory=diagrams,  transparent=True)

def computing_covariance(kernel, 
                         x, 
                         formula,
                         stub,
                         prec='1.2',
                         diagrams='../slides/diagrams/kern'):
    """
    Visualize covariance computation.

    :param kernel: Kernel function.
    :param x: Input data.
    :param formula: Formula to display.
    :param stub: Stub parameter.
    :param prec: Precision for values (default: '1.2').
    :param diagrams: Directory to save the plots (default: '../slides/diagrams/kern').
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)
    counter=0
    fig, ax = plt.subplots(1, 2, figsize=one_figsize)
    if len(x)>3:
        nsf=2
    base_text = ''
    data_text = ''
    for i, val in enumerate(x.flatten()):
        data_text += '$x_{i}={val:{prec}}$'.format(i=i, val=val,prec=prec)
        if i<len(x.flatten())-2:
            data_text += ', '
        elif i<len(x.flatten())-1:
            data_text += ' and '
    param_text = ''
    for i, param in enumerate(kernel.parameters):
        if param in notation_map:
            param_text += '$' + notation_map[param] 
        else:
            param_text += '$\\text{' + param + '}' 
        param_text += '={param:{prec}}$'.format(param=kernel.parameters[param],prec=prec)
        if i<len(kernel.parameters)-2:
            param_text += ', '
        elif i<len(kernel.parameters)-1:
            param_text += ' and '
            
    base_text += data_text
    if len(kernel.parameters)>0:
        base_text += ' with ' + param_text
    ax[0].set_position([0.0, 0.0, 1.0, 1.0])
    ax[0].set(ylim=[0.0, 1.0])
    ax[0].set(xlim=[0.0, 1.0])
    clear_axes(ax[0])
    
    ax[0].text(0.5, 0.9, formula, ha='center', fontsize=20)
    ax[0].text(0.5, 0.2, base_text, ha='center', fontsize=12)
    ax[1].set_position([0.5, 0.3, 0.5, 0.5])
    clear_axes(ax[1])
    K = kernel.K(x)
    KplotFull = np.array(K, dtype='str')
    Kplot = KplotFull.copy()
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            KplotFull[i, j] = '{value:{prec}}'.format(value=K[i, j], prec=prec)
            Kplot[i, j] = ''
    base_file_name = 'computing_{stub}_covariance'.format(stub=stub)
    a = ax[0].text(0.25, 0.6, '', ha='center', fontsize=16)
    for j in range(K.shape[0]):
        for i in range(j+1):
            text = '$x_{i}={val_i:{prec}}$, $x_{j}={val_j:{prec}}$'.format(i=i, j=j, val_i=x[i,0], val_j=x[j, 0], prec=prec)
            a.set_text(text)
            #a.append(ax[0].text(0.25, 0.4, 
            #                    ['$\\kernelScalar_{' num2str(i) ', ' num2str(j) '} = ' numsf2str(variance, nsf) ' \\times \\exp \\left(-\\frac{(' numsf2str(t(i), nsf) '-' numsf2str(t(j), nsf) ')^2}{2\\times ' numsf2str(lengthScale, nsf) '^2}\\right)$'], 'horizontalalignment', 'center')])
            file_name = base_file_name+'{counter:0>3}.svg'.format(counter=counter)
            ma.write_figure(file_name, directory=diagrams, transparent=True)
            counter += 1
            Kplot[i, j] = KplotFull[i, j]

            matrix(Kplot, 
                   ax=ax[1], 
                   bracket_style='square', 
                   type='entries',
                   highlight=True,
                   highlight_row = [i, i],
                   highlight_col = [j, j],
                   highlight_color=[1, 0, 1])
            
            file_name = base_file_name+'{counter:0>3}.svg'.format(counter=counter)
            ma.write_figure(file_name, directory=diagrams, 
                              transparent=True)
            counter +=1

            if i != j:
                Kplot[j, i] = KplotFull[j, i]

                matrix(Kplot, 
                       ax=ax[1], 
                       bracket_style='square', 
                       type='entries',
                       highlight=True,
                       highlight_row = [j, j],
                       highlight_col = [i, i],
                       highlight_color=[1, 0, 1])
                file_name = base_file_name+'{counter:0>3}.svg'.format(counter=counter)
                ma.write_figure(file_name, directory=diagrams,
                                  transparent=True)
                counter += 1

    matrix(Kplot, 
           ax=ax[1], 
           bracket_style='square', 
           type='entries')

    file_name = base_file_name+'{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams,
                      transparent=True)
    counter += 1

    matrix(K, 
           ax=ax[1], 
           bracket_style='square',
           type='image')
    file_name = base_file_name+'{counter:0>3}.svg'.format(counter=counter)
    ma.write_figure(file_name, directory=diagrams, 
                      transparent=True)
    counter += 1
            
def kern_circular_sample(K, mu=None, x=None,
                         filename=None, fig=None, num_samps=5,
                         num_theta=48, multiple=True,
                         diagrams='../diagrams', **kwargs):
    """
    Sample from a circular kernel and create animation.

    :param K: Kernel function.
    :param mu: Mean (optional).
    :param x: Input data (optional).
    :param filename: Output filename (optional).
    :param fig: Matplotlib figure (optional).
    :param num_samps: Number of samples (default: 5).
    :param num_theta: Number of theta values (default: 48).
    :param multiple: Whether to show multiple samples (default: True).
    :param diagrams: Directory to save the plots (default: '../diagrams').
    :param **kwargs: Additional keyword arguments.
    :returns: Animation object.
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)

    if x is None:
        if multiple:
            n=K.shape[0]/num_samps
        x = np.linspace(-1, 1, n)[:, np.newaxis]
        
        if multiple:
            x = output_augment_x(x, num_samps)

    else:
        n=x.shape[0]

    
    if multiple:
        R1 = np.random.normal(size=(n*num_samps,1))
        R2 = np.random.normal(size=(n*num_samps,1))
    else:
        R1 = np.random.normal(size=(n, num_samps))
        R2 = np.random.normal(size=(n, num_samps))
        
    U1 = np.dot(R1,np.diag(1/np.sqrt(np.sum(R1*R1, axis=0))))
    R2 = R2 - np.dot(U1,np.diag(np.sum(R2*U1, axis=0)))
    R2 = np.dot(R2,np.diag(np.sqrt(np.sum(R1*R1, axis=0))/np.sqrt(np.sum(R2*R2, axis=0))))
    L = np.linalg.cholesky(K+np.diag(np.ones((K.shape[0])))*1e-6)

    LR1 = np.dot(L,R1)
    LR2 = np.dot(L,R2)
    

    from matplotlib import animation
    if multiple:
        x_lim = (np.min(x[:, 1]), np.max(x[:, 1]))
    else:
        x_lim = (np.min(x.flatten()), np.max(x.flatten()))
    
    y_lim = np.sqrt(2)*np.array([np.min(np.array([np.min(LR1.flatten()), np.min(LR2.flatten())])),
                        np.max(np.array([np.max(LR1.flatten()), np.max(LR2.flatten())]))])
    
    if fig is None:
        fig, _ = plt.subplots(figsize=one_figsize)
    rect = 0, 0, 1., 1.
    ax = fig.add_axes(rect)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    line = []
    for i in range(num_samps):
        l, = ax.plot([], [], lw=2)
        line.append(l)
        
    # initialization function: plot the background of each frame
    def init():
        for i in range(num_samps):
            line[i].set_data([], [])
        return line

    # animation function.  This is called sequentially
    def animate(i):
        theta = float(i)/num_theta*tau
        xc = np.cos(theta)
        yc = np.sin(theta)
        # generate 2d basis in t-d space
        coord = xc*R1 + yc*R2
        y = xc*LR1 + yc*LR2
        if mu is not None:
            y = y + mu
        if multiple:
            end = 0
        for j in range(num_samps):
            if multiple:
                start = end
                end += n
                line[j].set_data(x[start:end, 1], y[start:end, 0])
            else:
                line[j].set_data(x, y[:, j])
        return line

    # call the animator.  blit=True means only re-draw the parts that have changed.
    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_theta, blit=True)

def animate_covariance_function(kernel_function,
                                x=None, num_samps=5,
                                multiple=False):
    """
    Create animation of covariance function samples.

    :param kernel_function: Kernel function to sample from.
    :param x: Input data (optional).
    :param num_samps: Number of samples (default: 5).
    :param multiple: Whether to show multiple samples (default: False).
    :returns: Animation object.
    """

    fig, ax = plt.subplots(figsize=one_figsize)

    if x is None:
        n=200
        x = np.linspace(-1, 1, n)[:, np.newaxis]
    
    if multiple:
        x = output_augment_x(x, num_samps)
    
    K = kernel_function(x, x)
    return K, kern_circular_sample(K, x=x,
                                   fig=fig, num_samps=num_samps,
                                   multiple=multiple)
    
def covariance_func(kernel, x=None,
                    shortname=None, longname=None, comment=None,
                    num_samps=5, diagrams='../diagrams', multiple=False):
    """
    Plot covariance function samples.

    :param kernel: Kernel function to sample from.
    :param x: Input data (optional).
    :param shortname: Short name for the kernel (optional).
    :param longname: Long name for the kernel (optional).
    :param comment: Comment to display (optional).
    :param num_samps: Number of samples (default: 5).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    :param multiple: Whether to show multiple samples (default: False).
    """
    if not os.path.exists(diagrams):
        os.mkdir(diagrams)
    K, anim=animate_covariance_function(kernel.K, x, num_samps,
                                        multiple)

    if kernel.shortname is not None:
        filename = kernel.shortname + '_covariance'
    else:
        filename = 'covariance'

    ma.write_animation(anim,
                         filename + '.gif',
                         directory=diagrams,
                         writer='imagemagick',
                         fps=30)


    K2 = kernel.K(x[::10, :])
    fig, ax = plt.subplots(figsize=one_figsize)
    hcolor = [1., 0., 1.]
    obj = matrix(K2, ax=ax, type='image',
                 bracket_style='boxes', colormap='gray')

    ma.write_figure(filename + '.svg', directory=diagrams, transparent=True)

    if kernel.name is not None:
        out = '<h2>' + kernel.name + ' Covariance</h2>'
        out += '\\n\n'
    else:
        out = ''
    if kernel.formula is not None:
        out += '<p><center>' + kernel.formula + '</center></p>'
    out += '<table>\\n  <tr><td><img src="' + ma.filename_join(filename, diagrams) + '.svg"></td><td><img src="' + ma.filename_join(filename, diagrams) + '.gif"></td></tr>\\n</table>'
    if comment is not None:
        out += '<p><center>' + comment + '</center></p>'
    fhand = open(ma.filename_join(filename + '.html', diagrams), 'w')
    fhand.write(out)

    
def rejection_samples(kernel, x=None, num_few=20, num_many=1000,  diagrams='../diagrams', **kwargs):
    """
    Generate rejection samples from a kernel.

    :param kernel: Kernel function to sample from.
    :param x: Input data (optional).
    :param num_few: Number of few samples (default: 20).
    :param num_many: Number of many samples (default: 1000).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    :param **kwargs: Additional keyword arguments.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=big_wide_figsize)
    if x is None:
        x = np.linspace(-1, 1, 250)[:, np.newaxis]
    resolution = x.shape[0]
    K = kernel.K(x, x, **kwargs)
    f = np.random.multivariate_normal(np.zeros(resolution), K, size=num_few).T
    #ax.set_xticks(range(1, 26, 2))
    #ax.set_yticks([-1, 0, 1])
    ylim = [-4, 4]
    xlim = [np.min(x.flatten()), np.max(x.flatten() )]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_position([0., 0., 1., 1.])
    ax.set_axis_off()
    h_f = ax.plot(x, f)
    ma.write_figure('gp_rejection_sample001.png', directory=diagrams, transparent=True)

    fnew = np.random.multivariate_normal(np.zeros(resolution), K, size=num_many-num_few).T
    f = np.hstack((f, fnew))
    h_f += ax.plot(x, fnew)
    ma.write_figure('gp_rejection_sample002.png', directory=diagrams, transparent=True)

    ind = [int(resolution/5.), int(2*resolution/3.), int(4*resolution/5.)]
    K_data = K[ind][:, ind]
    x_data = x[ind, :]
    y_data = np.random.multivariate_normal(np.zeros(len(ind)), K_data, size=1).T
    
    h_data=ax.plot(x_data, y_data, 'o', markersize=25, linewidth=3, color=[0., 0., 0.])
    ma.write_figure('gp_rejection_sample003.png', directory=diagrams, transparent=True)
    delta = y_data - f[ind, :]
    dist = (delta*delta).sum(0)
    del_ind = np.argsort(dist)[10:]
    for i in del_ind:
        h_f[i].remove()
    ma.write_figure('gp_rejection_sample004.png', directory=diagrams, transparent=True)

    # This is not the numerically stable way to do this!
    Kinv = np.linalg.inv(K_data)
    Kinvy = np.dot(Kinv, y_data)
    K_star = kernel.K(x_data, x, **kwargs)
    A = np.dot(Kinv, K_star)
    mu_f = np.dot(A.T, y_data)
    c_f = np.diag(K - np.dot(A.T, K_star))[:, np.newaxis]
    if GPY_AVAILABLE:
        import mlai.gp_tutorial as gpt
        _ = gpt.gpplot(x,
                           mu_f,
                           mu_f-2*np.sqrt(c_f),
                           mu_f+2*np.sqrt(c_f), 
                           ax=ax)
    else:
        ax.plot(x, mu_f, 'b-', linewidth=2)
        ax.fill_between(x.flatten(), (mu_f-2*np.sqrt(c_f)).flatten(), (mu_f+2*np.sqrt(c_f)).flatten(), alpha=0.3)
    ma.write_figure('gp_rejection_sample005.png', directory=diagrams, transparent=True)
    
    
def two_point_sample(kernel_function, diagrams='../diagrams'):
    """
    Sample from a two-point kernel function.

    :param kernel_function: Kernel function to sample from.
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    ind = [0, 1]    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=two_figsize)
    x = np.linspace(-1, 1, 25)[:, np.newaxis]
    K = kernel_function(x, x)
    obj = matrix(K, ax=ax[1], type='image', colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    #fig.colorbar(mappable=obj, ax=ax[1])
    #ax[1].set_axis('off')
    ma.write_figure('two_point_sample000.svg', directory=diagrams, transparent=True)

    f = np.random.multivariate_normal(np.zeros(25), K, size=1)
    ax[0].plot(range(1, 26), f.flatten(), 'o', markersize=5, linewidth=3, color=[1., 0., 0.])
    ax[0].set_xticks(range(1, 26, 2))
    ax[0].set_yticks([-1, 0, 1])
    ylim = [-1.5, 1.5]
    xlim = [0, 26]
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].set_xlabel('$i$', fontsize=20)
    ax[0].set_ylabel('$f$', fontsize=20)
    ma.write_figure('two_point_sample001.svg', directory=diagrams, transparent=True)

    ax[0].plot(np.array(ind)+1, [f[0,ind[0]], f[0,ind[1]]], 'o', markersize=10, linewidth=5, color=hcolor)
    ma.write_figure('two_point_sample002.svg', directory=diagrams, transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                 highlight=True, 
                 highlight_row=[0, 1], 
                 highlight_col=[0,1], 
                 highlight_color=hcolor,
                 colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample003.svg', directory=diagrams, transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                 highlight=True, 
                 highlight_row=[0, 1], 
                 highlight_col=[0,1], 
                 highlight_color=hcolor,
                 highlight_width=5,
                 zoom=True,
                 zoom_row=[0, 9],
                 zoom_col=[0, 9],
                 colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample004.svg', directory=diagrams, transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                 highlight=True, 
                 highlight_row=[0, 1], 
                 highlight_col=[0,1], 
                 highlight_color=hcolor,
                 highlight_width=6,
                 zoom=True,
                 zoom_row=[0, 4],
                 zoom_col=[0, 4],
                 colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample005.svg', directory=diagrams, transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                 highlight=True, 
                 highlight_row=[0, 1], 
                 highlight_col=[0,1], 
                 highlight_color=hcolor,
                 highlight_width=7,
                 zoom=True,
                 zoom_row=[0, 2],
                 zoom_col=[0, 2],
                 colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample006.svg', directory=diagrams, transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                 highlight=True, 
                 highlight_row=[0, 1], 
                 highlight_col=[0,1], 
                 highlight_color=hcolor,
                 highlight_width=8,
                 zoom=True,
                 zoom_row=[0, 1],
                 zoom_col=[0, 1],
                 colormap='gray')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample007.svg', directory=diagrams, transparent=True)

    obj = matrix(K[ind][:, ind], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample008.svg', directory=diagrams, transparent=True)

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=ind, stub='two_point_sample', start=9, diagrams=diagrams)

    ind = [0, 7]
    ax[0].cla()
    ax[0].set_aspect('auto')
    ax[0].plot(range(1, 26), f.flatten(), 'o', markersize=5, linewidth=3, color=[1., 0., 0.])
    ax[0].set_xticks(range(1, 26, 2))
    ax[0].set_yticks([-1, 0, 1])
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].set_xlabel('$i$', fontsize=20)
    ax[0].set_ylabel('$f$', fontsize=20)
    
    ax[0].plot(np.array(ind)+1, [f[0,ind[0]], f[0,ind[1]]], 'o', markersize=10, linewidth=5, color=hcolor)
    obj = matrix(K[ind][:, ind], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\\prime$',fontsize=16)
    ma.write_figure('two_point_sample013.svg', directory=diagrams, transparent=True)

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=ind, stub='two_point_sample', start=14, diagrams=diagrams)


def poisson(diagrams='../diagrams'):
    """
    Plot Poisson distribution examples.

    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    from scipy.stats import poisson
    fig, ax = plt.subplots(figsize=two_figsize)
    y = np.asarray(range(0, 16))
    p1 = poisson.pmf(y, mu=1.)
    p3 = poisson.pmf(y, mu=3.)
    p10 = poisson.pmf(y, mu=10.)

    ax.plot(y, p1, 'r.-', markersize=20, label='$\\lambda=1$', lw=3)
    ax.plot(y, p3, 'g.-', markersize=20, label='$\\lambda=3$', lw=3)
    ax.plot(y, p10, 'b.-', markersize=20, label='$\\lambda=10$', lw=3)
    ax.set_title('Poisson Distribution', fontsize=20)
    ax.set_xlabel('$y_i$', fontsize=20)
    ax.set_ylabel('$p(y_i)$', fontsize=20)
    ax.legend(fontsize=20)
    ma.write_figure('poisson.svg', directory=diagrams, transparent=True)

def logistic(diagrams='../diagrams'):
    """
    Plot logistic function examples.

    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    fig, ax = plt.subplots(figsize=two_figsize)
    f = np.linspace(-8, 8, 100)
    g = 1/(1+np.exp(-f))
    
    ax.plot(f, g, 'r-', lw=3)
    ax.set_title('Logistic Function', fontsize=20)
    ax.set_xlabel('$f_i$', fontsize=20)
    ax.set_ylabel('$g_i$', fontsize=20)
    ma.write_figure('logistic.svg', directory=diagrams, transparent=True)


def height(ax, h, ph):
    """Plot height as a distribution."""
    ax.plot(h, ph, '-', color=[1, 0, 0], linewidth=3)
    ax.set_xticks([1.25, 1.7, 2.15])
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel('$h/m$', fontsize=20)
    ax.set_ylabel('$p(h)$', fontsize=20)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    #ax.vlines(xlim[0], ylim[0], ylim[1], color='k')
    #ax.hlines(ylim[0], xlim[0], xlim[1], color='k')

def weight(ax, w, pw):
    """
    Plot weight distribution.

    :param ax: Matplotlib axis.
    :param w: Weight values.
    :param pw: Weight probabilities.
    """
    ax.plot(w, pw, '-', color=[0, 0, 1.], linewidth=3)
    ax.set_xticks([55, 75, 95])
    ax.set_yticks([0.02, 0.04, 0.06])
    ax.set_xlabel('$w/kg$', fontsize=20)
    ax.set_ylabel('$p(w)$', fontsize=20)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    #ax.vlines(xlim[0], ylim[0], ylim[1], color='k')
    #ax.hlines(ylim[0], xlim[0], xlim[1], color='k')

def low_rank_approximation(fontsize=25, diagrams='../diagrams'):
    """
    Visualize low-rank matrix approximation.

    :param fontsize: Font size for labels (default: 25).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    fig, ax = plt.subplots(1, 4, figsize=big_wide_figsize)
    q = 3
    k1 = 10
    k2 = 12
    blank_canvas(ax[3])
    ax[3].text(0.145, 0.55, r'$\\times$', 
               horizontalalignment='center',
               fontsize=fontsize)
    ax[3].text(0.47, 0.55, r'$=$', 
               horizontalalignment='center',
               fontsize=fontsize)
    ax[3].text(0.075, 0.55, r'$\\mathbf{U}$', 
               horizontalalignment='center',
               fontsize=fontsize, color=[1, 1, 1])
    ax[3].text(0.3, 0.55, r'$\\mathbf{V}^\\top$', 
               horizontalalignment='center',
               fontsize=fontsize, color=[1, 1, 1])
    ax[3].text(0.65, 0.55, r'$\\mathbf{W}$', 
               horizontalalignment='center',
               fontsize=fontsize, color=[1, 1, 1])
    U = np.random.randn(k1, q)
    VT = np.random.randn(q, k2)
    basewidth = 0.15
    ax[0].set_position([0.0, 0.15, basewidth, basewidth/q*k1])
    matrix(U, ax=ax[0], type='image')
    ax[1].set_position([0.0, 0.5, basewidth/q*k2, basewidth])
    ax[1].set_aspect('equal')
    matrix(VT, ax=ax[1], type='image')
    ax[2].set_position([0.35, 0.15, basewidth/q*k2, basewidth/q*k1])
    matrix(np.dot(U,VT), ax=ax[2], type='image')
    ax[3].set_frame_on(True)
    ax[3].axes.get_yaxis().set_visible(True)
    ma.write_figure('wisuvt.svg', directory=diagrams, transparent=True)
    
def kronecker_illustrate(fontsize=25, diagrams='../diagrams'):
    """
    Illustrate Kronecker product concept.

    :param fontsize: Font size for labels (default: 25).
    :param diagrams: Directory to save the plot (default: '../diagrams').
    """
    fig, ax = plt.subplots(1, 4, figsize=two_figsize)
    A = [['$a$', '$b$'],
         [ '$c$', '$d$']]
    B = [['$\\mathbf{K}$']]

    AkroneckerB = [['$a\\mathbf{K}$', '$b\\mathbf{K}$'],
                    ['$c\\mathbf{K}$', '$d\\mathbf{K}$']]
    ax[0].set_position([0, 0, 1, 1])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].text(0.4, 0.5, ' $\\otimes$', horizontalalignment='center',
                  fontsize=fontsize)
    ax[0].text(0.55, 0.5, ' $=$', horizontalalignment='center',
                  fontsize=fontsize)

    ax[1].set_position([0.15, 0.4, 0.2, 0.2])
    objA = matrix(A, ax=ax[1], bracket_style='square', type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.45, 0.45, 0.05, 0.1])
    objB = matrix(B, ax=ax[2], bracket_style='none', type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.57, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3],
                    bracket_style='square',
                    type='entries',
                  fontsize=fontsize)
    ax[0].set_axis_off()
        
    ma.write_figure('kronecker_product.svg', directory=diagrams, transparent=True)
def blank_canvas(ax):
    """
    Create a blank canvas for plotting.

    :param ax: Matplotlib axis to clear.
    """
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def kronecker_illustrate(fontsize=25, figsize=two_figsize, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    A = [['$a$', '$b$'],
         [ '$c$', '$d$']]
    B = [['$\\mathbf{K}$']]

    AkroneckerB = [['$a\\mathbf{K}$', '$b\\mathbf{K}$'],
                    ['$c\\mathbf{K}$', '$d\\mathbf{K}$']]

    blank_canvas(ax[0])
    ax[0].text(0.4, 0.5, ' $\\otimes$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.55, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.15, 0.4, 0.2, 0.2])
    objA = matrix(A, ax=ax[1], bracket_style='square', type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.45, 0.45, 0.05, 0.1])
    objB = matrix(B, ax=ax[2], bracket_style='none', type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.57, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3], bracket_style='square', type='entries',
                  fontsize=fontsize)
        
    ma.write_figure('kronecker_illustrate.svg', directory=diagrams, transparent=True)

def kronecker_IK(fontsize=25, figsize=two_figsize, reverse=False, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    my_rgb = [[1., 1., 1.],[1., 0., 0.],[ 0., 1., 0.],[ 0., 0., 1.]]

    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(my_rgb, name='primary+black')
    dim_I = 3
    dim_K = 3
    I = np.eye(dim_I)
    L = np.tril(np.ones(dim_K))
    K = np.dot(L, L.T)
        
    blank_canvas(ax[0])
    ax[0].text(0.3, 0.5, ' $\\otimes$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.615, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.05, 0.05, 0.2, 0.9])
    objI = matrix(np.stack([1-I]*3, 2), ax=ax[1],
                  bracket_style='boxes', type='colorpatch',
                  fontsize=fontsize)


    ax[2].set_position([0.35, 0.05, 0.2, 0.9])
    objK = matrix(np.stack((K==1, K==2, K==3), 2),
                  ax=ax[2],
                  bracket_style='boxes', type='colorpatch',
                  fontsize=fontsize)
    if reverse:
        kron_IK = np.kron(K, I)    
    else:
        kron_IK = np.kron(I, K)    
    ax[3].set_position([0.675, 0.1, 0.3, 0.85])
    objAkB = matrix(np.stack((np.logical_or(kron_IK==1, kron_IK==0),
                              np.logical_or(kron_IK==2, kron_IK==0),
                              np.logical_or(kron_IK==3, kron_IK==0)), 2),
                    ax=ax[3],
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize)
    if reverse:
        ma.write_figure('kronecker_KI.svg', directory=diagrams, transparent=True)
    else:
        ma.write_figure('kronecker_IK.svg', directory=diagrams, transparent=True)

def kronecker_IK_highlight(fontsize=25, figsize=two_figsize, reverse=False, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    my_rgb = [[1., 1., 1.],[1., 0., 0.],[ 0., 1., 0.],[ 0., 0., 1.]]
    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(my_rgb, name='primary+black')
    dim_I = 3
    dim_K = 3
    I = np.eye(dim_I)
    L = np.tril(np.ones(dim_K))
    K = np.dot(L, L.T)
    if reverse:
        kron_IK = np.kron(K, I)
        stem = 'KI'
    else:
        kron_IK = np.kron(I, K)
        stem = 'IK'
        
    IK_stack = np.stack((np.logical_or(kron_IK==1, kron_IK==0),
                         np.logical_or(kron_IK==2, kron_IK==0),
                         np.logical_or(kron_IK==3, kron_IK==0)), 2)        
    ax.set_position([0, 0, 1, 1])
    objAkB = matrix(IK_stack,
                    ax=ax,
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize)
        
    ma.write_figure('kronecker_{stem}_highlighted001.svg'.format(stem=stem), directory=diagrams)
    objAkB = matrix(IK_stack,
                    ax=ax,
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize,
                    highlight=True, 
                    highlight_row=[0, 2], 
                    highlight_col=[0, 2], 
                    highlight_color=hcolor,
                    highlight_width=8)
    ma.write_figure('kronecker_{stem}_highlighted002.svg'.format(stem=stem), directory=diagrams)
    count = 2
    for zoom in [6, 3, 2]:
        objAkB = matrix(IK_stack,
                        ax=ax,
                        bracket_style='boxes', type='colorpatch',
                        fontsize=fontsize,
                        highlight=True, 
                        highlight_row=[0, 2], 
                        highlight_col=[0, 2], 
                        highlight_color=hcolor,
                        highlight_width=8,
                        zoom=True,
                        zoom_row=[0, zoom],
                        zoom_col=[0, zoom])
        count+=1
        ma.write_figure('kronecker_{stem}_highlighted{count:0>3}.svg'.format(stem=stem, count=count), directory=diagrams)

def kronecker_WX(fontsize=25, figsize=two_figsize, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    A = [['$\\mathbf{W}$', '$\\mathbf{0}$', '$\\mathbf{0}$'],['$\\mathbf{0}$', '$\\mathbf{W}$', '$\\mathbf{0}$'],['$\\mathbf{0}$', '$\\mathbf{0}$', '$\\mathbf{W}$']]
    B = [['$\\mathbf{x}_{1,:}$'],['$\\mathbf{x}_{2,:}$'],['$\\mathbf{x}_{3,:}$']]
    AkroneckerB = [['$\\mathbf{W}\\mathbf{x}_{1,:}$'],[ '$\\mathbf{W}\\mathbf{x}_{2,:}$'], ['$\\mathbf{W}\\mathbf{x}_{3,:}$']]

    blank_canvas(ax[0])
    ax[0].text(0.4, 0.5, r'$\\times$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.65, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.05, 0.35, 0.3, 0.3])
    objA = matrix(A, ax=ax[1], bracket_style='square',
                  type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.4, 0.35, 0.25, 0.3])
    objB = matrix(B, ax=ax[2], bracket_style='none',
                  type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.6, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3], bracket_style='square',
                    type='entries',
                    fontsize=fontsize)
        
    ma.write_figure('kronecker_WX.svg', directory=diagrams,
                      transparent=True)

def perceptron(x_plus, x_minus, learn_rate=0.1, max_iters=10000,
               max_updates=30, seed=100001, diagrams='../diagrams'):
    """Fit a perceptron algorithm and record iterations of fit"""
    w, b, x_select = ma.init_perceptron(x_plus, x_minus, seed=seed)
    updates = 0
    count = 0
    iterations = 0
    setup=True
    f2, ax2 = plt.subplots(1, 2, figsize=two_figsize)
    handle = init_perceptron(f2, ax2, x_plus, x_minus, w, b)
    handle['plane'].set_visible(False)
    handle['arrow'].set_visible(False)
    handle['circle'] = plt.Circle((x_select[0], x_select[1]), 0.25, color='b', fill=False)
    ax2[0].add_artist(handle['circle'])
    ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.svg'.format(samp=count), directory=diagrams, transparent=True)
    extent = ax2[0].get_window_extent().transformed(f2.dpi_scale_trans.inverted())
    ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.png'.format(samp=count), directory=diagrams, bbox_inches=extent, transparent=True)
    count += 1
    handle['plane'].set_visible(True)
    handle['arrow'].set_visible(True)
    ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.svg'.format(samp=count), directory=diagrams, transparent=True)
    ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.png'.format(samp=count), directory=diagrams, bbox_inches=extent, transparent=True)

    while updates<max_updates and iterations<max_iters:
        iterations += 1
        w, b, x_select, updated = ma.update_perceptron(w, b, x_plus, x_minus, learn_rate)
        if updated:
            updates += 1
            count+=1
            handle['circle'].center = x_select[0], x_select[1]
            ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.svg'.format(samp=count), directory=diagrams, transparent=True)     
            ma.write_figure(figure=f2, filename='perceptron{samp:0>3}.png'.format(samp=count), bbox_inches=extent, directory=diagrams, transparent=True)        
            count+=1
            handle = update_perceptron(handle, f2, ax2, x_plus, x_minus, updates, w, b)
            ma.write_figure(filename='perceptron{samp:0>3}.svg'.format(samp=count),
                              figure=f2,
                              directory=diagrams,
                              transparent=True)
            ma.write_figure(filename='perceptron{samp:0>3}.png'.format(samp=count),
                              figure=f2, 
                              directory=diagrams,
                              bbox_inches=extent,
                              transparent=True)
    print('Data passes:', iterations)
    return count



def dist2(X, Y):
    """Computer squared distances between two design matrices"""
    return -2*np.dot(X,Y.T) + (X*X).sum(1)[:, np.newaxis] + (Y*Y).T.sum(0)

def clear_axes(ax):
    """Clear the axes lines and ticks"""
    ax.tick_params(axis='both',          
                   which='both', 
                   bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False) 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

def non_linear_difficulty_plot_3(alpha=1.0,
                                 rbf_width=2,
                                 num_basis_func=3,
                                 num_samples=10,
                                 number_across=30,
                                 fontsize=30,
                                 diagrams='../diagrams'):
    """Push a Gaussian density through an RBF network and plot results"""

    mu = np.linspace(-4, 4, num_basis_func)[np.newaxis, :]
    W = np.random.randn(num_samples, num_basis_func)*np.sqrt(alpha)

    x1 = np.linspace(-1, 1, number_across)
    x2 = x1; mu1 = mu; mu2 = mu
    MU1, MU2 = np.meshgrid(mu1, mu2)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.flatten(), X2.flatten()])

    MU = np.column_stack([MU1.flatten(), MU2.flatten()])
    num_basis_func = MU.shape[0]
    number = X.shape[0]
    Phi = np.exp(-dist2(X, MU)/(2*rbf_width*rbf_width))
    num_samples = 3
    np.random.seed(13)
    W = np.random.randn(num_samples, num_basis_func)*np.sqrt(alpha)
    F = np.dot(Phi,W.T)

    fig, ax = plt.subplots(1, 3, figsize=two_figsize)
    fig.delaxes(ax[2])
    ax[2] = fig.add_subplot(133, projection='3d')

    start_val = 0
    for i in range(number_across):
          end_val = number_across*(i+1)
          a = ax[0].plot(X[start_val:end_val, 0], X[start_val:end_val, 1], 'r-')
          start_val = end_val

    # Reshape X to plot lines in opposite directions
    X1 = X1.T
    X2 = X2.T
    X = np.column_stack([X1.flatten(), X2.flatten()])
    start_val = 0
    for i in range(number_across):
          end_val = number_across*(i+1)
          a = ax[0].plot(X[start_val:end_val, 0], X[start_val:end_val, 1], 'r-')
          start_val = end_val

    ax[0].tick_params(axis='both',          
        which='both', bottom=False, top=False, labelbottom=False,
                      right=False, left=False, labelleft=False) 
    ax[0].set(aspect='equal')
    clear_axes(ax[0])
    ax[0].set_xlabel('$x_1$', ha='center', fontsize=fontsize)
    ax[0].set_ylabel('$x_2$', ha='center', fontsize=fontsize)

    start_val = 0
    for i in range(number_across):
          end_val = number_across*(i+1)
          a = ax[2].plot(F[start_val:end_val, 0], 
                            F[start_val:end_val, 1], 
                            F[start_val:end_val, 2], 
                            'r-')
          start_val = end_val

    # Reshape F to plot lines in opposite directions
    F1 = np.reshape(F[:, 0], (X1.shape[0], X1.shape[1]),order='F')
    F2 = np.reshape(F[:, 1], (X1.shape[0], X1.shape[1]),order='F')
    F3 = np.reshape(F[:, 2], (X1.shape[0], X1.shape[1]),order='F')
    F = np.column_stack([F1.flatten(), F2.flatten(), F3.flatten()])

    start_val = 0
    for i in range(number_across):
          end_val = number_across*(i+1)
          if True:
                a = ax[2].plot(F[start_val:end_val, 0], 
                            F[start_val:end_val, 1], 
                            F[start_val:end_val, 2],
                            'r-')
          start_val = end_val

    # Treble axis size to increase plot size
    fig.delaxes(ax[1])
    ax[1] = fig.add_subplot(132)
    pos = ax[2].get_position()
    scale=2.5
    npos = [0, 0,  pos.width*scale, pos.height*scale] 
    npos[0] = pos.x0 - 0.5*(npos[2] - pos.width)
    npos[1] = pos.y0 - 0.5*(npos[3] - pos.height)
    ax[2].set_position(npos)
    ax[2].set_axis_off()

    # Axis for writing text on plot
    ax[1].set(position=[0, 0, 1, 1])
    ax[1].set(xlim=[0, 1])
    ax[1].set(ylim=[0, 1])
    ax[1].set_axis_off()
    ax[1].text(0.5, 0.55, '$y_j = f_j(\\mathbf{x})$', 
               ha='center',
              fontsize=fontsize)
    ax[1].text(0.5, 0.45, '$\\longrightarrow$', 
               ha='center',
               fontsize=4*fontsize/3)
    ma.write_figure("nonlinear-mapping-3d-plot.svg",
                      directory=diagrams,
                      figure=fig,
                      transparent=True)

def non_linear_difficulty_plot_2(alpha=1.0,
                                 rbf_width=2,
                                 num_basis_func=3,
                                 num_samples=10,
                                 number_across=101,
                                 fontsize=30,
                                 diagrams='../diagrams'):
    """Plot a one dimensional line mapped through a two dimensional mapping."""
    fig, ax = plt.subplots(1, 3, figsize=two_figsize)
    for item in ax:
        item.patch.set_visible(False)

    W = np.random.randn(num_samples, num_basis_func)*np.sqrt(alpha)

    x = np.linspace(-6, 6, number_across)[:, np.newaxis]
    mu = np.linspace(-4, 4, num_basis_func)[np.newaxis, :]
    number = x.shape[0]
    Phi = np.exp(-dist2(x, mu.T)/(2*rbf_width*rbf_width))
    
    F = np.dot(Phi,W.T)

    a = ax[0].plot(x, np.ones(x.shape), 'r-')
    subx = x[0::10,:]
    b = ax[0].plot(subx, np.ones(subx.shape), 'b.')
    ax[0].set(ylim=[0.5, 1.5])
    ax[0].set(Xlim=[-7, 7])
    ax[0].set(aspect='equal')
    clear_axes(ax[0])
    a[0].set(linewidth=3)
    b[0].set(markersize=20)

    ax[0].set_xlabel('$x$', ha='center', fontsize=fontsize)



    a = ax[2].plot(F[:, 0], F[:, 1], 'r-')
    b = ax[2].plot(F[0::10][:,0], F[0::10][:,1], 'b.')
    a[0].set(linewidth=3)
    b[0].set(markersize=20)
    ax[2].set(aspect='equal')
    clear_axes(ax[2])

    ax[2].set_xlabel('$y_1$', ha='center', fontsize=fontsize)
    ax[2].set_ylabel('$y_2$', ha='center', fontsize=fontsize)

    # Axis for writing text on plot
    ax[1].set(position=[0, 0, 1, 1])
    ax[1].set(xlim=[0, 1])
    ax[1].set(ylim=[0, 1])
    ax[1].set_axis_off()
    ax[1].text(0.5, 0.65, '$y_1 = f_1(x)$', ha='center', fontsize=fontsize)
    ax[1].text(0.5, 0.5, '$\\longrightarrow$', ha='center', fontsize=4*fontsize/3)
    ax[1].text(0.5, 0.35, '$y_2 = f_2(x)$', ha='center', fontsize=fontsize)
    ma.write_figure('nonlinear-mapping-2d-plot.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)

def non_linear_difficulty_plot_1(alpha=1.0,
                                 data_std=0.2,
                                 rbf_width=0.1,
                                 num_basis_func=100,
                                 number_across=200,
                                 num_samples=1000,
                                 patch_color = [0.3, 0.3, 0.3],
                                 fontsize=30,
                                 diagrams='../diagrams'):
    """Plot a one dimensional Gaussian pushed through an RBF network."""
    from matplotlib.patches import Polygon
    xsamp = np.random.randn(num_samples, 1)
    x = np.linspace(-6, 6, number_across)[:, np.newaxis]

    # Create RBF network with much larger variation in functions.
    mu = np.linspace(-4, 4, num_basis_func)[np.newaxis, :]
    Phi = np.exp(-dist2(xsamp, mu.T)/(2*rbf_width*rbf_width))
    W = np.random.randn(1, num_basis_func)*np.sqrt(alpha)
    f = np.dot(Phi,W.T)

    fig, ax = plt.subplots(1, 3, figsize=three_figsize)
    p = np.exp(-0.5/alpha*x**2)*1/np.sqrt(2*np.pi*alpha)
    patch = Polygon(np.column_stack((x, p)), closed=True, facecolor=patch_color)
    a = ax[0].add_patch(patch)
    a.set(linewidth=2)

    clear_axes(ax[0])
    ax[0].set(ylim=[0, 0.5])
    ax[0].set(xlim=[-6, 6])
    ax[0].set_xlabel('$p(x)$', ha='center', fontsize=20)

    y = np.linspace(np.min(f.flatten())-3*data_std, np.max(f.flatten())+3*data_std, 100)[:, np.newaxis]
    p = np.mean(np.exp(-0.5/(data_std*data_std)*dist2(y, f))*1/(np.sqrt(2*np.pi)*data_std), 1)
    patch = Polygon(np.column_stack((y, p)), closed=True, facecolor=patch_color)
    a=ax[2].add_patch(patch)
    a.set(linewidth=2)

    clear_axes(ax[2])
    ax[2].set(ylim=[0, 0.5])
    ax[2].set(xlim=[-6, 6])
    ax[2].set_xlabel('$p(y)$', ha='center', fontsize=20)
    
    # Axis for writing text on plot
    ax[1].set_position([0, 0, 1, 1])
    ax[1].set(xlim=[0, 1])
    ax[1].set(ylim=[0, 1])
    ax[1].set_axis_off()
    ax[1].text(0.5, 0.45, '$y = f(x) + \\epsilon$', ha='center', fontsize=fontsize)
    ax[1].text(0.5, 0.35, '$\\longrightarrow$', ha='center', fontsize=4*fontsize/3)
    ma.write_figure('gaussian-through-nonlinear.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)

class network():
    """Class for drawing a neural network."""

    def __init__(self, layers=None):
        if layers is None:
            self.layers=[]
        else:
            self.layers=layers

    def add_layer(self, layer):
        self.layers.append(layer)

    @property
    def width(self):
        """Return the widest layer number"""
        store = 0
        for layer in self.layers:
            if layer.width>store:
                store = layer.width
        return store

    @property
    def depth(self):
        """Return the depth of the network"""
        return len(self.layers)


    def draw(self, grid_unit=2.5, node_unit=0.9,
             observed_style='shaded', line_width=1,
             origin=[0,0]):
        """Draw the network using daft"""
        shape = [self.depth, self.width]
        xpadding = 2
        ypadding = 0
        pgm = daft.PGM(shape=[shape[0]+xpadding, shape[1]+ypadding],
                       origin=origin, 
                       grid_unit=grid_unit, 
                       node_unit=node_unit, 
                       observed_style=observed_style,
                       line_width=line_width)
        
        yoffset = 0.5
        for i, layer in enumerate(self.layers):
            posy = yoffset + i*(shape[1])/(self.depth)        
            for j in range(layer.width):
                xoffset = (shape[0])*(self.width-layer.width)/(2*(self.width))+0.5
                posx = xoffset + j*(shape[0])/(self.width)
                pgm.add_node(daft.Node(layer.label.format(index=j+1),
                                       ('$' + layer.label + '$').format(index=j+1), 
                                       posx, posy,
                                       observed=layer.observed,
                                       fixed=layer.fixed))
            for j in range(layer.width):
                if i > 0:
                    parent = self.layers[i-1]
                    for k in range(parent.width):
                            pgm.add_edge(parent.label.format(index=k+1), 
                                         layer.label.format(index=j+1))

        ctx = pgm.render()
        fig = ctx.figure
        ax = plt.gca()
        for i, layer in enumerate(self.layers):
            posy = yoffset + i*(shape[1]-ypadding)/(self.depth)
            posx = shape[0] + xpadding/2 
            x, y = pgm._ctx.convert(posx, posy)
            a = []
            a.append(ax.text(x, y, layer.text,
                             ha="center", va="center",
                             fontsize=20))
        return fig, ax



class layer():
    """Class for a neural network layer"""
    def __init__(self, width=5, label='', observed=False, fixed=False, text=''):
        self.width = width
        self.label = label
        self.observed = observed
        self.fixed = fixed
        self.text = text


def deep_nn(diagrams='../diagrams'):
    """Draw a deep neural network."""
    model = network()
    model.add_layer(layer(width=6, label='x_{index}',
                    observed=True, text=r'given $\\mathbf{x}$'))
    model.add_layer(layer(width=8, label='h_{{1, {index}}}',
                    text=r'$\\mathbf{h}_1=\\boldsymbol{\\phi}\\left(\\mathbf{W}_1\\mathbf{x}\\right)$'))
    model.add_layer(layer(width=6, label='h_{{2, {index}}}',
                    text=r'$\\mathbf{h}_2=\\boldsymbol{\\phi}\\left(\\mathbf{W}_2\\mathbf{h}_1\\right)$'))
    model.add_layer(layer(width=4, label='h_{{3, {index}}}',
                    text=r'$\\mathbf{h}_3=\\boldsymbol{\\phi}\\left(\\mathbf{W}_3\\mathbf{h}_2\\right)$'))
    model.add_layer(layer(width=1, label='y',
                    text=r'$y=\\mathbf{w}_4^\\top\\mathbf{h}_3$',
                    observed=True))
    fig, ax = model.draw()
    ma.write_figure('deep-nn2.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)

    new_text = ['', '', '', '', '']
    for i, text in enumerate(new_text):
        model.layers[i].text=text
    fig, ax = model.draw()
    ma.write_figure('deep-nn1.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)


    
def deep_nn_bottleneck(diagrams='../diagrams'):
    """Draw a deep neural network with bottleneck layers."""
    model = network()
    model.add_layer(layer(width=6, label='x_{index}',
                    observed=True, text=r'given $\\mathbf{x}$'))
    model.add_layer(layer(width=4, label='z_{{1, {index}}}',
                    fixed=True, text=r'$\\mathbf{z}_1 = \\mathbf{V}_1^\\top\\mathbf{x}$'))
    model.add_layer(layer(width=8, label='h_{{1, {index}}}',
                    text=r'$\\mathbf{h}_1=\\boldsymbol{\\phi}\\left(\\mathbf{U}_1\\mathbf{z}_1\\right)$'))
    model.add_layer(layer(width=4, label='z_{{2, {index}}}',
                    text=r'$\\mathbf{z}_2 = \\mathbf{V}_2^\\top\\mathbf{h}_1$',
                    fixed=True))
    model.add_layer(layer(width=6, label='h_{{2, {index}}}',
                    text=r'$\\mathbf{h}_2=\\boldsymbol{\\phi}\\left(\\mathbf{U}_2\\mathbf{z}_2\\right)$'))
    model.add_layer(layer(width=2, label='z_{{3, {index}}}',
                    text = r'$\\mathbf{z}_2 = \\mathbf{V}_3^\\top\\mathbf{h}_2$',
                    fixed=True))
    model.add_layer(layer(width=4, label='h_{{3, {index}}}',
                    text=r'$\\mathbf{h}_3=\\boldsymbol{\\phi}\\left(\\mathbf{U}_3\\mathbf{z}_3\\right)$'))
    model.add_layer(layer(width=1, label='y',
                    text=r'$y=\\mathbf{w}_4^\\top\\mathbf{h}_3$',
                    observed=True))
    fig, ax = model.draw()
    ma.write_figure('deep-nn-bottleneck2.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)

    new_text = ['input layer', 'latent layer 1', 'hidden layer 1', 
                'latent layer 2', 'hidden layer 2', 'latent layer 3', 
                'hidden layer 3', 'output layer']
    fig, ax = model.draw()
    ma.write_figure('deep-nn-bottleneck1.svg',
                      directory=diagrams,
                      figure=fig,
                      transparent=True)
    for i, text in enumerate(new_text):
        model.layers[i].text=text


def box(lim_val=0.5, side_length=25):
    """Plot a box for use in deep GP samples."""
    t = np.hstack((lim_val*np.ones((side_length, 1)), 
                   np.linspace(-lim_val, lim_val, side_length)[:, np.newaxis]))
    tnew = np.hstack((np.linspace(lim_val, -lim_val, side_length)[:, np.newaxis], 
                      lim_val*np.ones((side_length, 1))))
    t = np.vstack((t, tnew))
    tnew = np.hstack((-lim_val*np.ones((side_length, 1)), 
                      np.linspace(lim_val, -lim_val, side_length)[:, np.newaxis]))
    t = np.vstack((t, tnew))
    tnew = np.hstack((np.linspace(-lim_val, lim_val, side_length)[:, np.newaxis], 
                   -lim_val*np.ones((side_length, 1))))
    t = np.vstack((t, tnew))
    return t

def stack_gp_sample(kernel=None,
                    latent_dims=[2, 2, 2, 2, 2],
                    side_length=25, lim_val=0.5, num_samps=5,figsize=(1.4, 7),
                    diagrams='../diagrams'):
    """Draw a sample from a deep Gaussian process."""

    if kernel is None:
        try:
            import GPy
        except ImportError:
            print('GPy unavailable, see https://github.com/SheffieldML/GPy pip install GPy')
            return
        kernel=GPy.kern.RBF
    
    
    depth=len(latent_dims)
    num_time = side_length*4
    t = box(lim_val=lim_val, side_length=side_length)
    fig, ax = plt.subplots(len(latent_dims), 1, figsize=figsize)

    for i in range(num_samps):
        X = []
        X.append(t)
        kern = []
        K = []
        for j in range(depth):
            kern.append(kernel(X[j].shape[1]))
            K.append(kern[j].K(X[j]))
            X.append(np.random.multivariate_normal(mean=np.zeros((num_time)), 
                                                     cov=K[j], size=latent_dims[j]).T)


        for j in range(depth):
            #pos = ax[j].get_position()
            #ax[j].set_position((pos.x0, pos.y0, 
            #                    pos.width/2, pos.height))
            if j == 0:
                ax[j].set(xlim=1.25*np.array([-lim_val, lim_val]))
                ax[j].set(ylim=1.25*np.array([-lim_val, lim_val]))
            else: 
                ax[j].set(xlim=1.25*np.array([-1, 1]))
                ax[j].set(ylim=1.25*np.array([-1, 1]))
            ax[j].cla()
            ax[j].plot(X[j][:, 0], X[j][:, 1], color='b', linewidth=2)
            ax[j].set(aspect="equal")
            ax[j].set_axis_off()
        file_name = 'stack-gp-sample-' + kern[0].name + '-' + str(i) + '.svg'
        ma.write_figure(file_name,
                          directory=diagrams,
                          figure=fig,
                          transparent=True)

        if False:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            for j in range(2):
                pos = ax[j].get_position()
                ax[j].set_position((pos.x0, pos.y0, pos.width/2, pos.height))
                if j == 0:
                    ax[j].set(xlim=1.25*np.array([-lim_val, lim_val]))
                    ax[j].set(ylim=1.25*np.array([-lim_val, lim_val]))
                else: 
                    ax[j].set(xlim=1.25*np.array([-1, 1]))
                    ax[j].set(ylim=1.25*np.array([-1, 1]))
                if j == 1:
                    plt.plot(X[0][:, 0], X[0][:, 1], 
                             color='b', linewidth=2)
                else:
                    plt.plot(X[-1][:, 0], 
                             X[-1][:, 1], 
                             color='b', linewidth=2)
                ax[j].set_axis_off()
            file_name = 'stack-gp-sample-squash-' + str(i) + '.svg'
            ma.write_figure(file_name,
                              directory=diagrams,
                              figure=fig,
                              transparent=True)

def vertical_chain(depth=5, grid_unit=1.5, node_unit=1, line_width=1.5, shape=None, target='y'):
    """Make a verticle chain representation of a deep GP"""
    if shape is None:
        shape = [node_unit, 2*node_unit+depth]
    direction = [0, -node_unit]

    pgm = daft.PGM(shape=shape,
                   origin=[0, 0], 
                   grid_unit=grid_unit, 
                   node_unit=node_unit, 
                   observed_style='shaded',
                  line_width=line_width)

    node = "x"
    pgm.add_node(daft.Node("x", r"$\\mathbf{x}$", 0.5, 6.5, fixed=True))
    for i in range(depth):
        last = node
        node="f_{index}".format(index=i+1)
        pgm.add_node(daft.Node(node, r"$\\mathbf{{f}}_{index}$".format(index=i+1),
                               0.5, depth-i + 0.5))
        pgm.add_edge(last, node)

    last = node
    node = target
    pgm.add_node(daft.Node(node, r"$\\mathbf{y}$", 
                           0.5, 0.5, observed=True))
    pgm.add_edge(last, node)
    return pgm

def horizontal_chain(depth=5,
                     shape=None,
                     origin=[0, 0],
                     grid_unit=4,
                     node_unit=1.9,
                     line_width=3,
                    target="y"):
    """Plot a horizontal Markov chain."""
    if shape is None:
        shape = [2*node_unit+depth, node_unit]
    
    direction = [-node_unit, 0]
    
    pgm = daft.PGM(shape=[7, 1],
                   origin=origin, 
                   grid_unit=grid_unit, 
                   node_unit=node_unit, 
                   observed_style='shaded',
                  line_width=line_width)

    node = "x"
    pgm.add_node(daft.Node(node, r"$\\mathbf{x}$", 0.5, 0.5, fixed=True))
    for i in range(depth):
        last=node
        node = "f_{index}".format(index=i+1)
        pgm.add_node(daft.Node(node, r"$\\mathbf{{f}}_{index}$".format(index=i+1), 
                               i+1.5, 0.5))
        pgm.add_edge(last, node)
    last = node
    node=target
    pgm.add_node(daft.Node(target, r"$\\mathbf{y}$", depth+1.5, 0.5, observed=True))
    pgm.add_edge(last, node)
    return pgm

def shared_gplvm():
    """Plot graphical model of a Shared GP-LVM"""
    pgm = daft.PGM(shape=[4, 3],
                   origin=[0, 0], 
                   grid_unit=5, 
                   node_unit=1.9, 
                   observed_style='shaded',
                  line_width=3)

    pgm.add_node(daft.Node("t", r"$\\mathbf{t}$", 2, 2.5, observed=True))
    pgm.add_node(daft.Node("X", r"$\\mathbf{X}$", 2, 1.5))
    pgm.add_node(daft.Node("Z_1", r"$\\mathbf{Z}_1$", 1, 1.5))
    pgm.add_node(daft.Node("Z_2", r"$\\mathbf{Z}_2$", 3, 1.5))
    pgm.add_node(daft.Node("Y_1", r"$\\mathbf{Y}_1$", 1.5, 0.5, observed=True))
    pgm.add_node(daft.Node("Y_2", r"$\\mathbf{Y}_2$", 2.5, 0.5, observed=True))
    pgm.add_edge("t", "X")
    pgm.add_edge("X", "Y_1")
    pgm.add_edge("X", "Y_2")
    pgm.add_edge("Z_1", "Y_1")
    pgm.add_edge("Z_2", "Y_2")
    return pgm

def three_pillars_innovation(diagrams='./diagrams'):
    """Plot graphical model of three pillars of successful innovation"""
    pgm = daft.PGM(shape=[4, 2.5],
                   origin=[0, 0], 
                   grid_unit=5, 
                   node_unit=4.2, 
                   observed_style='shaded',
                   line_width=6)
    import matplotlib
    orig_font_size = matplotlib.rcParams['font.size']
    orig_font_weight = matplotlib.rcParams['font.weight']
    matplotlib.rc('font', size=32)
    matplotlib.rc('font', weight='bold')
    aspect=2
    pgm.add_node(daft.Node("innovate", "innovate", 2, 1.75, aspect=aspect))
    ax=pgm.render()
    ma.write_figure('three-pillars-innovation001.svg',
                      directory=diagrams,
                      figure=ax.figure,
                      transparent=True)
    pgm.add_node(daft.Node("resolve", "resolve", 3, 0.75, aspect=aspect))
    pgm.add_edge("resolve", "innovate", directed=False)
    ax=pgm.render()
    ma.write_figure('three-pillars-innovation002.svg',
                      directory=diagrams,
                      figure=ax.figure,
                      transparent=True)
    pgm.add_node(daft.Node("deploy", "deploy", 1, 0.75, aspect=aspect))
    pgm.add_edge("innovate", "deploy", directed=False)
    pgm.add_edge("deploy", "resolve", directed=False)
    ax=pgm.render()
    ma.write_figure('three-pillars-innovation003.svg',
                      directory=diagrams,
                      figure=ax.figure,
                      transparent=True)
    matplotlib.rc('font', size=orig_font_size)
    matplotlib.rc('font', weight=orig_font_weight)


def model_output(model, output_dim=0, scale=1.0, offset=0.0, ax=None, xlabel='$x$', ylabel='$y$', xlim=None, ylim=None, fontsize=20, portion=0.2):
    """Plot the output of a GP.
    :param model: the model for the output plotting.
    :param output_dim: the output dimension to plot.
    :param scale: how to scale the output.
    :param offset: how to offset the output.
    :param ax: axis to plot on.
    :param xlabel: label for the x axis (default: '$x$').
    :param ylabel: label for the y axis (default: '$y$').
    :param xlim: limits of the x axis
    :param ylim: limits of the y axis
    :param fontsize: fontsize (default 20)
    :param portion: What proportion of the input range to put outside the data."""
    if ax is None:
        fig, ax = plt.subplots(figsize=big_figsize)
    ax.plot(model.X.flatten(), model.Y[:, output_dim]*scale + offset, 'r.',markersize=10)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    xt = pred_range(model.X, portion=portion)
    if xlim is None:
        xlim = [np.min(xt.flatten()), np.max(xt.flatten())]

    yt_mean, yt_var = model.predict(xt)
    yt_mean = yt_mean*scale + offset
    yt_var *= scale*scale
    yt_sd = np.sqrt(yt_var)
    if yt_sd.shape[1]>1:
        yt_sd = yt_sd[:, output_dim]

    if GPY_AVAILABLE:
        import mlai.gp_tutorial as gpt
        _ = gpt.gpplot(xt.flatten(),
                           yt_mean[:, output_dim],
                           yt_mean[:, output_dim]-2*yt_sd.flatten(),
                           yt_mean[:, output_dim]+2*yt_sd.flatten(), 
                           ax=ax)
    else:
        ax.plot(xt.flatten(), yt_mean[:, output_dim], 'b-', linewidth=2)
        ax.fill_between(xt.flatten(), 
                       (yt_mean[:, output_dim]-2*yt_sd.flatten()).flatten(), 
                       (yt_mean[:, output_dim]+2*yt_sd.flatten()).flatten(), 
                       alpha=0.3)


    if ylim is None:
        ylim=ax.get_ylim()
    else:
        ax.set_ylim(ylim)
    
    if hasattr(model, 'Z'):
        z = model.Z.flatten()
        ax.plot(z, np.full_like(z, ylim[0]), 'k^', markersize='20')

    ax.autoscale(enable=True, axis='x', tight=True)
    return ax

def model_sample(model, output_dim=0, scale=1.0, offset=0.0,
                 samps=10, ax=None, xlabel='$x$', ylabel='$y$', 
                 fontsize=20, portion=0.2,
                 xlim=None, ylim=None):
    """Plot model output with samples."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=big_figsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    xt = pred_range(model.X, portion=portion)
    yt_mean, yt_var = model.predict(xt)
    yt_mean = yt_mean*scale + offset
    yt_var *= scale*scale
    yt_sd=np.sqrt(yt_var)
    if yt_sd.shape[1]>1:
        yt_sd = yt_sd[:, output_dim]
        
    if GPY_AVAILABLE:
        import mlai.gp_tutorial as gpt
        _ = gpt.gpplot(xt.flatten(),
                   yt_mean[:, output_dim],
                   yt_mean[:, output_dim]-2*yt_sd.flatten(),
                   yt_mean[:, output_dim]+2*yt_sd.flatten(), 
                   ax=ax)
    else:
        ax.plot(xt.flatten(), yt_mean[:, output_dim], 'b-', linewidth=2)
        ax.fill_between(xt.flatten(), 
                       (yt_mean[:, output_dim]-2*yt_sd.flatten()).flatten(), 
                       (yt_mean[:, output_dim]+2*yt_sd.flatten()).flatten(), 
                       alpha=0.3)
    for i in range(samps):
        xt = pred_range(model.X, portion=portion, randomize=True)
        a = model.posterior_sample(xt)
        ax.plot(xt.flatten(), a[:, output_dim]*scale+offset, 'b.',markersize=3, alpha=0.2)
    ax.plot(model.X.flatten(), model.Y[:, output_dim]*scale+offset, 'r.',markersize=10)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([np.min(xt.flatten()), np.max(xt.flatten())])
    if ylim is not None: 
        ax.set_ylim(ylim)
               
    if hasattr(model, 'Z'):
        ylim = ax.get_ylim()
        ax.plot(model.Z, np.ones(model.Z.shape)*ax.get_ylim()[0], marker='^', linestyle=None, markersize=20)
    return ax

def multiple_optima(ax=None, gene_number=937, resolution=80, model_restarts=10, seed=10000, max_iters=300, optimize=True, fontsize=20, diagrams='./diagrams'):
    """
    Show an example of a multimodal error surface for Gaussian process
    regression. Gene 937 has bimodal behaviour where the noisy mode is
    higher.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=one_figsize)

    ylim = [-1, 5]
    xlim = [10, 50]
    # Contour over a range of length scales and signal/noise ratios.
    log_SNRs = np.linspace(ylim[0], ylim[1], resolution)
    length_scales = np.linspace(xlim[0], xlim[1], resolution)

    try:
        import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.della_gatta_TRP63_gene_expression(data_set='della_gatta',gene_number=gene_number)

    y = data['Y']
    x = data['X']
    offset = y.mean()
    scale = np.sqrt(y.var())

    yhat = (y-offset)/scale

    try:
        import GPy
    except ImportError:
        print('GPy unavailable, see https://github.com/SheffieldML/GPy pip install GPy')
        return
    kernel = GPy.kern.RBF(1, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(x, yhat, kernel=kernel)
    lls = ma.contour_data(model, data, length_scales, log_SNRs)
    ax.contour(length_scales, log_SNRs, np.exp(lls), 20, cmap=plt.cm.jet)
    #ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('length scale', fontsize=fontsize)
    ax.set_ylabel('$\\log_{10}$ SNR', fontsize=fontsize)

    ma.write_figure('multiple-optima000.svg',
                      directory=diagrams,
                      figure=ax.figure,
                      transparent=True)
    

    # Now run a few optimizations
    models = []
    optim_point_x = np.empty(2)
    optim_point_y = np.empty(2)

    np.random.seed(seed=seed)
    noises = [1., 1., 0.001]
    lengthscales = [50., 2000., 20] 
    for noise, lengthscale in zip(noises, lengthscales):
        kern = GPy.kern.RBF(1, lengthscale=lengthscale)

        m = GPy.models.GPRegression(x, yhat, kernel=kern)
        m.likelihood.variance = noise
        optim_point_x[0] = m.rbf.lengthscale
        optim_point_y[0] = np.log10(m.rbf.variance) - np.log10(m.likelihood.variance);

        # optimize
        if optimize:
            _ = m.optimize()

        optim_point_x[1] = m.rbf.lengthscale
        optim_point_y[1] = np.log10(m.rbf.variance) - np.log10(m.likelihood.variance);

        from matplotlib.patches import Arrow
        ax.arrow(optim_point_x[0],
                 optim_point_y[0],
                 optim_point_x[1] - optim_point_x[0],
                 optim_point_y[1] - optim_point_y[0],
                 head_length=1,
                 head_width=0.5, fc='k', ec='k')
        models.append(m)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ma.write_figure('multiple-optima001.svg',
                      directory=diagrams,
                      figure=ax.figure,
                      transparent=True)
    
    return m, lls 


# def rotate_object(rotation_matrix, handles):
#     """Rotate an object in an image"""
#     for i in handles:
# 	if type(handle) is text:
#             handle.get('position')
#             xy[0:1] = np.dot(rotation_matrix,xy[0:1].T)
#             handle.set('position', xy)
#         else:
#             xd = handle.get('xdata')
#             yd = handle.get('ydata')
#             new = np.dot(rotation_matrix,np.column_stack((xd[:].T, yd[:].T)))
#             handle.set('xdata', new[0, :])
#             handle.set('ydata', new[1, :])

def google_trends(terms, initials, diagrams='./diagrams'):
    """Plot google trends data for a number of different terms."""
    import pods
    import matplotlib.dates as mdates
    data = pods.datasets.google_trends(terms)
    data['data frame'].set_index('Date', inplace=True)
    fig, ax = plt.subplots(figsize=wide_figsize)
    data['data frame'].plot(ax=ax, rot=45)
    ma.write_figure(initials+'-google-trends.svg',
                      directory=diagrams,
                      transparent=True)

    handles = ax.get_lines()
    for handle in handles:
        handle.set_visible(False)
    for i, handle in enumerate(handles):
        handle.set_visible(True)
        ma.write_figure('{initials}-google-trends{sample:0>3}.svg'.format(initials=initials,sample=i),
                          directory=diagrams,
                          transparent=True)
    return ax
                         


