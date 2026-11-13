Linear Regression Tutorial
==========================

Linear regression is the foundation of supervised learning, modeling the relationship between input features and continuous target variables. This tutorial explores the mathematical foundations, implementation, and practical usage of linear regression using MLAI.

Mathematical Background
-----------------------

Linear regression models the relationship between input features :math:`\mathbf{x}` and target variable :math:`y` as:

.. math::

   y = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) + \epsilon

where :math:`\boldsymbol{\phi}(\mathbf{x})` are basis functions, :math:`\mathbf{w}` are the model weights, and :math:`\epsilon` is the noise term.

The goal is to find the optimal weights that minimize the sum of squared errors:

.. math::

   \mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_i))^2

This can be solved analytically using the normal equation:

.. math::

   \mathbf{w}^* = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{y}

where :math:`\boldsymbol{\Phi}` is the design matrix.

Implementation in MLAI
---------------------

MLAI provides a comprehensive implementation of linear regression. Let's explore how to use it:

.. code-block:: python

   import numpy as np
   import mlai.mlai as mlai
   import matplotlib.pyplot as plt
   import pandas as pd

   # Generate synthetic regression data
   np.random.seed(42)
   n_samples = 100
   n_features = 1
   
   # Create non-linear data with noise
   x_data = np.linspace(-3, 3, n_samples).reshape(-1, 1)
   y_data = 2 * x_data**2 - 3 * x_data + 1 + 0.5 * np.random.randn(n_samples, 1)
   
   print(f"Data shape: {x_data.shape}")
   print(f"Target shape: {y_data.shape}")

Creating the Model
------------------

Let's create a linear regression model with polynomial basis functions:

.. code-block:: python

   # Create polynomial basis functions
   basis = mlai.Basis(mlai.polynomial, 3, data_limits=[x_data.min(), x_data.max()])
   
   # Create linear model
   model = mlai.LM(x_data, y_data, basis)
   
   print(f"Model created with {basis.number} basis functions")
   print(f"Basis matrix shape: {basis.Phi(x_data).shape}")

Training the Model
-----------------

Now let's train the model and examine the results:

.. code-block:: python

   # Fit the model
   model.fit()
   
   # Get the learned weights
   print(f"Learned weights: {model.w_star}")
   
   # Compute predictions
   y_pred = model.predict(x_data)
   
   # Calculate R-squared
   ss_res = np.sum((y_data - y_pred) ** 2)
   ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
   r_squared = 1 - (ss_res / ss_tot)
   print(f"R-squared: {r_squared:.3f}")

Visualizing the Results
----------------------

Let's visualize the data, model fit, and basis functions:

.. code-block:: python

   def plot_linear_regression_results(x_data, y_data, model, basis):
       """Plot the data, model fit, and basis functions."""
       plt.figure(figsize=(15, 5))
       
       # Plot 1: Data and model fit
       plt.subplot(1, 3, 1)
       plt.scatter(x_data, y_data, c='blue', alpha=0.6, label='Data')
       
       # Create smooth curve for model prediction
       x_smooth = np.linspace(x_data.min(), x_data.max(), 200).reshape(-1, 1)
       y_smooth = model.predict(x_smooth)
       plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Model Fit')
       
       plt.xlabel('x')
       plt.ylabel('y')
       plt.title('Linear Regression with Polynomial Basis')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Plot 2: Basis functions
       plt.subplot(1, 3, 2)
       Phi = basis.Phi(x_smooth)
       for i in range(Phi.shape[1]):
           plt.plot(x_smooth, Phi[:, i], label=f'Basis {i+1}', linewidth=2)
       
       plt.xlabel('x')
       plt.ylabel('Basis Function Value')
       plt.title('Basis Functions')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Plot 3: Residuals
       plt.subplot(1, 3, 3)
       residuals = y_data.flatten() - y_pred.flatten()
       plt.scatter(y_pred, residuals, c='green', alpha=0.6)
       plt.axhline(y=0, color='red', linestyle='--')
       plt.xlabel('Predicted Values')
       plt.ylabel('Residuals')
       plt.title('Residual Plot')
       plt.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

   # Plot the results
   plot_linear_regression_results(x_data, y_data, model, basis)

Model Evaluation
----------------

Let's evaluate the model performance using various metrics:

.. code-block:: python

   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

   # Compute predictions
   y_pred = model.predict(x_data)
   
   # Calculate metrics
   mse = mean_squared_error(y_data, y_pred)
   mae = mean_absolute_error(y_data, y_pred)
   r2 = r2_score(y_data, y_pred)
   
   print("Model Performance Metrics:")
   print(f"Mean Squared Error: {mse:.4f}")
   print(f"Mean Absolute Error: {mae:.4f}")
   print(f"R-squared: {r2:.4f}")
   print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

Regularization
--------------

Let's explore how regularization affects the model:

.. code-block:: python

   # Create models with different regularization strengths
   regularization_strengths = [0, 0.1, 1.0, 10.0]
   models = {}
   
   for alpha in regularization_strengths:
       # Create model with regularization
       model_reg = mlai.LM(x_data, y_data, basis, alpha=alpha)
       model_reg.fit()
       
       # Evaluate
       y_pred_reg = model_reg.predict(x_data)
       mse_reg = mean_squared_error(y_data, y_pred_reg)
       r2_reg = r2_score(y_data, y_pred_reg)
       
       models[alpha] = {
           'model': model_reg,
           'mse': mse_reg,
           'r2': r2_reg,
           'weights': model_reg.w_star
       }
       
       print(f"Alpha = {alpha}: MSE = {mse_reg:.4f}, R² = {r2_reg:.4f}")
   
   # Plot regularization effects
   plt.figure(figsize=(15, 5))
   
   # Plot 1: Model fits
   plt.subplot(1, 3, 1)
   plt.scatter(x_data, y_data, c='blue', alpha=0.6, label='Data')
   
   x_smooth = np.linspace(x_data.min(), x_data.max(), 200).reshape(-1, 1)
   for alpha in regularization_strengths:
       y_smooth = models[alpha]['model'].predict(x_smooth)
       plt.plot(x_smooth, y_smooth, label=f'α = {alpha}', linewidth=2)
   
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('Effect of Regularization')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Plot 2: Weight magnitudes
   plt.subplot(1, 3, 2)
   for alpha in regularization_strengths:
       weights = models[alpha]['weights']
       plt.plot(range(len(weights)), np.abs(weights), 
                marker='o', label=f'α = {alpha}', linewidth=2)
   
   plt.xlabel('Weight Index')
   plt.ylabel('|Weight|')
   plt.title('Weight Magnitudes')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Plot 3: Performance metrics
   plt.subplot(1, 3, 3)
   alphas = list(models.keys())
   mses = [models[alpha]['mse'] for alpha in alphas]
   r2s = [models[alpha]['r2'] for alpha in alphas]
   
   plt.plot(alphas, mses, 'bo-', label='MSE', linewidth=2)
   plt.xlabel('Regularization Strength (α)')
   plt.ylabel('Mean Squared Error')
   plt.title('Regularization vs Performance')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Comparing Different Basis Functions
----------------------------------

Let's compare how different basis functions perform on the same data:

.. code-block:: python

   # Test different basis functions
   basis_configs = [
       ('Linear', mlai.linear, 1),
       ('Polynomial', mlai.polynomial, 3),
       ('Radial', mlai.radial, 5),
       ('Fourier', mlai.fourier, 4)
   ]
   
   results = {}
   
   for name, basis_func, num_basis in basis_configs:
       # Create basis and model
       basis = mlai.Basis(basis_func, num_basis, data_limits=[x_data.min(), x_data.max()])
       model = mlai.LM(x_data, y_data, basis)
       model.fit()
       
       # Evaluate
       y_pred = model.predict(x_data)
       mse = mean_squared_error(y_data, y_pred)
       r2 = r2_score(y_data, y_pred)
       
       results[name] = {
           'mse': mse,
           'r2': r2,
           'model': model,
           'basis': basis
       }
       
       print(f"{name} basis: MSE = {mse:.4f}, R² = {r2:.4f}")
   
   # Plot comparison
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   axes = axes.ravel()
   
   for i, (name, result) in enumerate(results.items()):
       model = result['model']
       
       # Plot data and fit
       axes[i].scatter(x_data, y_data, c='blue', alpha=0.6, s=20)
       
       x_smooth = np.linspace(x_data.min(), x_data.max(), 200).reshape(-1, 1)
       y_smooth = model.predict(x_smooth)
       axes[i].plot(x_smooth, y_smooth, 'r-', linewidth=2)
       
       axes[i].set_title(f'{name} (MSE: {result["mse"]:.4f}, R²: {result["r2"]:.3f})')
       axes[i].set_xlabel('x')
       axes[i].set_ylabel('y')
       axes[i].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Bayesian Linear Regression
-------------------------

Let's explore Bayesian linear regression for uncertainty quantification:

.. code-block:: python

   # Create Bayesian linear model
   blm = mlai.BLM(x_data, y_data, basis)
   blm.fit()
   
   # Make predictions with uncertainty
   x_test = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
   y_pred_mean, y_pred_var = blm.predict(x_test)
   y_pred_std = np.sqrt(y_pred_var)
   
   # Plot Bayesian predictions
   plt.figure(figsize=(12, 8))
   
   # Plot data
   plt.scatter(x_data, y_data, c='blue', alpha=0.6, label='Data')
   
   # Plot mean prediction
   plt.plot(x_test, y_pred_mean, 'r-', linewidth=2, label='Mean Prediction')
   
   # Plot uncertainty bands
   plt.fill_between(x_test.flatten(), 
                   y_pred_mean.flatten() - 2*y_pred_std.flatten(),
                   y_pred_mean.flatten() + 2*y_pred_std.flatten(),
                   alpha=0.3, color='red', label='95% Confidence Interval')
   
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('Bayesian Linear Regression with Uncertainty')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Key Concepts
------------

1. **Least Squares**: Linear regression minimizes the sum of squared errors between predictions and targets.

2. **Basis Functions**: Transform input features to capture non-linear relationships while maintaining linearity in parameters.

3. **Regularization**: Prevents overfitting by penalizing large weights.

4. **Uncertainty Quantification**: Bayesian approaches provide uncertainty estimates for predictions.

Advantages and Limitations
-------------------------

**Advantages:**
- Simple and interpretable
- Fast training and prediction
- Provides uncertainty estimates (Bayesian)
- Works well with small datasets

**Limitations:**
- Assumes linear relationship in basis space
- May underperform on highly non-linear problems
- Sensitive to outliers
- Requires feature engineering for complex patterns

Further Reading
---------------

- `Linear Regression <https://en.wikipedia.org/wiki/Linear_regression>`_ on Wikipedia
- `Ordinary Least Squares <https://en.wikipedia.org/wiki/Ordinary_least_squares>`_
- `Ridge Regression <https://en.wikipedia.org/wiki/Ridge_regression>`_ for regularization
- `Bayesian Linear Regression <https://en.wikipedia.org/wiki/Bayesian_linear_regression>`_

This tutorial demonstrates how linear regression provides a solid foundation for understanding supervised learning, with extensions to handle non-linear relationships and uncertainty quantification. 