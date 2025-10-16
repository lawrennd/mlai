Basis Functions Tutorial
========================

Basis functions are a fundamental concept in machine learning that allow us to transform input data into a higher-dimensional space where linear models can capture complex, non-linear relationships. This tutorial explores the theory and practical implementation of basis functions using MLAI.

Mathematical Background
-----------------------

Instead of working directly on the original input space :math:`\mathbf{x}`, we build models in a new space :math:`\boldsymbol{\phi}(\mathbf{x})` where :math:`\boldsymbol{\phi}(\cdot)` is a vector-valued function defined on the input space.

For a one-dimensional input :math:`x`, a quadratic basis function is defined as:

.. math::

   \boldsymbol{\phi}(x) = \begin{bmatrix} 1 \\ x \\ x^2 \end{bmatrix}

This transforms a single input value into a 3-dimensional feature vector.

Types of Basis Functions in MLAI
--------------------------------

MLAI provides several types of basis functions. Let's explore them:

Linear Basis
~~~~~~~~~~~~

The simplest basis function that doesn't transform the input:

.. code-block:: python

   import numpy as np
   import mlai.mlai as mlai
   import matplotlib.pyplot as plt

   # Create linear basis
   x = np.linspace(-3, 3, 100).reshape(-1, 1)
   basis = mlai.Basis(mlai.linear, 1)
   Phi = basis.Phi(x)
   
   print(f"Input shape: {x.shape}")
   print(f"Basis output shape: {Phi.shape}")
   print(f"First few basis values:\n{Phi[:5]}")

Polynomial Basis
~~~~~~~~~~~~~~~~

Polynomial basis functions capture non-linear relationships:

.. code-block:: python

   # Create polynomial basis with 4 functions
   basis = mlai.Basis(mlai.polynomial, 4, data_limits=[-3, 3])
   Phi = basis.Phi(x)
   
   # Plot the basis functions
   plt.figure(figsize=(12, 8))
   for i in range(Phi.shape[1]):
       plt.plot(x, Phi[:, i], label=f'Basis {i+1}', linewidth=2)
   
   plt.xlabel('Input x')
   plt.ylabel('Basis Function Value')
   plt.title('Polynomial Basis Functions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Radial Basis Functions (RBF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Radial basis functions are centered around specific points and are useful for capturing local patterns:

.. code-block:: python

   # Create radial basis functions
   basis = mlai.Basis(mlai.radial, 5, data_limits=[-3, 3])
   Phi = basis.Phi(x)
   
   # Plot the basis functions
   plt.figure(figsize=(12, 8))
   for i in range(Phi.shape[1]):
       plt.plot(x, Phi[:, i], label=f'RBF {i+1}', linewidth=2)
   
   plt.xlabel('Input x')
   plt.ylabel('Basis Function Value')
   plt.title('Radial Basis Functions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Fourier Basis
~~~~~~~~~~~~~

Fourier basis functions are useful for capturing periodic patterns:

.. code-block:: python

   # Create Fourier basis functions
   basis = mlai.Basis(mlai.fourier, 6, data_limits=[-3, 3])
   Phi = basis.Phi(x)
   
   # Plot the basis functions
   plt.figure(figsize=(12, 8))
   for i in range(Phi.shape[1]):
       plt.plot(x, Phi[:, i], label=f'Fourier {i+1}', linewidth=2)
   
   plt.xlabel('Input x')
   plt.ylabel('Basis Function Value')
   plt.title('Fourier Basis Functions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

ReLU Basis
~~~~~~~~~~

Rectified Linear Unit (ReLU) basis functions introduce non-linearity through piecewise linear functions:

.. code-block:: python

   # Create ReLU basis functions
   basis = mlai.Basis(mlai.relu, 4, data_limits=[-3, 3])
   Phi = basis.Phi(x)
   
   # Plot the basis functions
   plt.figure(figsize=(12, 8))
   for i in range(Phi.shape[1]):
       plt.plot(x, Phi[:, i], label=f'ReLU {i+1}', linewidth=2)
   
   plt.xlabel('Input x')
   plt.ylabel('Basis Function Value')
   plt.title('ReLU Basis Functions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Using Basis Functions with Linear Models
----------------------------------------

Basis functions are most powerful when combined with linear models. Let's see how to use them:

.. code-block:: python

   # Generate some non-linear data
   np.random.seed(42)
   x_data = np.linspace(-3, 3, 50).reshape(-1, 1)
   y_data = 2 * x_data**2 - 3 * x_data + 1 + 0.5 * np.random.randn(50, 1)
   
   # Create polynomial basis
   basis = mlai.Basis(mlai.polynomial, 3, data_limits=[-3, 3])
   
   # Create linear model with basis functions
   model = mlai.LM(x_data, y_data, basis)
   
   # Fit the model
   model.fit()
   
   # Make predictions
   x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
   y_pred = model.predict(x_test)
   
   # Plot results
   plt.figure(figsize=(10, 8))
   plt.scatter(x_data, y_data, c='blue', alpha=0.6, label='Data')
   plt.plot(x_test, y_pred, 'r-', linewidth=2, label='Polynomial Fit')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('Polynomial Regression with Basis Functions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Comparing Different Basis Functions
----------------------------------

Let's compare how different basis functions perform on the same data:

.. code-block:: python

   # Test different basis functions
   basis_types = [
       ('Linear', mlai.linear, 1),
       ('Polynomial', mlai.polynomial, 3),
       ('Radial', mlai.radial, 5),
       ('Fourier', mlai.fourier, 6)
   ]
   
   plt.figure(figsize=(15, 10))
   
   for i, (name, basis_func, num_basis) in enumerate(basis_types):
       plt.subplot(2, 2, i+1)
       
       # Create basis and model
       basis = mlai.Basis(basis_func, num_basis, data_limits=[-3, 3])
       model = mlai.LM(x_data, y_data, basis)
       model.fit()
       
       # Make predictions
       y_pred = model.predict(x_test)
       
       # Plot
       plt.scatter(x_data, y_data, c='blue', alpha=0.6, s=20)
       plt.plot(x_test, y_pred, 'r-', linewidth=2)
       plt.title(f'{name} Basis Functions')
       plt.xlabel('x')
       plt.ylabel('y')
       plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Key Concepts
------------

1. **Feature Transformation**: Basis functions transform input data into a higher-dimensional space where linear relationships can capture non-linear patterns.

2. **Kernel Trick**: Some basis functions can be computed efficiently using the kernel trick, avoiding explicit computation of the high-dimensional features.

3. **Overfitting**: Using too many basis functions can lead to overfitting. Regularization techniques help control this.

4. **Bias-Variance Trade-off**: More complex basis functions reduce bias but increase variance.

Choosing the Right Basis Function
--------------------------------

- **Linear**: When you expect a linear relationship
- **Polynomial**: For smooth, continuous non-linear relationships
- **Radial**: For local patterns and clustering-like behavior
- **Fourier**: For periodic or oscillatory patterns
- **ReLU**: For piecewise linear relationships (common in neural networks)

Limitations and Considerations
-----------------------------

- **Curse of Dimensionality**: High-dimensional basis expansions can be computationally expensive
- **Feature Selection**: Not all basis functions may be relevant for a given problem
- **Interpretability**: Complex basis functions can make models harder to interpret
- **Hyperparameter Tuning**: The number and type of basis functions are hyperparameters that need tuning

Further Reading
---------------

- `Basis Function <https://en.wikipedia.org/wiki/Basis_function>`_ on Wikipedia
- `Kernel Methods <https://en.wikipedia.org/wiki/Kernel_method>`_ for efficient computation
- `Feature Engineering <https://en.wikipedia.org/wiki/Feature_engineering>`_ for practical applications

This tutorial demonstrates how basis functions enable linear models to capture complex, non-linear relationships in data, forming the foundation for many advanced machine learning techniques. 