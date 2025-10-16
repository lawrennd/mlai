Logistic Regression Tutorial
============================

Logistic regression is a fundamental classification algorithm that extends linear regression to handle binary classification problems. This tutorial explores the mathematical foundations, implementation, and practical usage of logistic regression using MLAI.

Mathematical Background
-----------------------

Logistic regression models the probability of a binary outcome using the logistic function. Given input features :math:`\mathbf{x}` and binary labels :math:`y \in \{0, 1\}`, the model predicts:

.. math::

   P(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})}}

where :math:`\boldsymbol{\phi}(\mathbf{x})` are basis functions and :math:`\mathbf{w}` are the model weights.

The logistic function (also called sigmoid) maps any real number to the interval :math:`[0, 1]`, making it suitable for probability modeling.

Log-Odds and Link Function
--------------------------

The log-odds (logit) is defined as:

.. math::

   \log \frac{P(y = 1 | \mathbf{x})}{P(y = 0 | \mathbf{x})} = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})

This is the link function that connects the linear predictor to the probability. The inverse link function is the logistic function.

Implementation in MLAI
---------------------

MLAI provides a comprehensive implementation of logistic regression. Let's explore how to use it:

.. code-block:: python

   import numpy as np
   import mlai.mlai as mlai
   import matplotlib.pyplot as plt
   import pandas as pd

   # Generate synthetic classification data
   np.random.seed(42)
   n_samples = 200
   n_features = 2
   
   # Create two classes with some overlap
   X_class0 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], n_samples//2)
   X_class1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n_samples//2)
   
   X = np.vstack([X_class0, X_class1])
   y = pd.Series(np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]))
   
   print(f"Data shape: {X.shape}")
   print(f"Class distribution: {y.value_counts()}")

Creating the Model
------------------

Let's create a logistic regression model with polynomial basis functions:

.. code-block:: python

   # Create polynomial basis functions
   basis = mlai.Basis(mlai.polynomial, 3, data_limits=[X.min(), X.max()])
   
   # Create logistic regression model
   model = mlai.LR(X, y, basis)
   
   print(f"Model created with {basis.number} basis functions")
   print(f"Basis matrix shape: {basis.Phi(X).shape}")

Training the Model
-----------------

Now let's train the model and examine the results:

.. code-block:: python

   # Fit the model
   model.fit()
   
   # Get the learned weights
   print(f"Learned weights: {model.w_star}")
   
   # Compute predictions
   y_pred_proba, Phi = model.predict(X)
   y_pred = (y_pred_proba > 0.5).astype(int)
   
   # Calculate accuracy
   accuracy = (y_pred == y).mean()
   print(f"Training accuracy: {accuracy:.3f}")

Visualizing the Results
----------------------

Let's visualize the decision boundary and probability contours:

.. code-block:: python

   def plot_logistic_results(X, y, model, basis):
       """Plot the data points, decision boundary, and probability contours."""
       plt.figure(figsize=(15, 5))
       
       # Create grid for visualization
       x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
       y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
       xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
       
       # Get predictions for grid points
       grid_points = np.c_[xx.ravel(), yy.ravel()]
       proba_grid, _ = model.predict(grid_points)
       proba_grid = proba_grid.reshape(xx.shape)
       
       # Plot 1: Data points and decision boundary
       plt.subplot(1, 3, 1)
       plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', alpha=0.6, label='Class 0')
       plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', alpha=0.6, label='Class 1')
       plt.contour(xx, yy, proba_grid, levels=[0.5], colors='black', linewidths=2)
       plt.xlabel('Feature 1')
       plt.ylabel('Feature 2')
       plt.title('Decision Boundary')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Plot 2: Probability contours
       plt.subplot(1, 3, 2)
       contour = plt.contourf(xx, yy, proba_grid, levels=20, cmap='RdYlBu')
       plt.colorbar(contour, label='P(y=1)')
       plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', alpha=0.6, s=20)
       plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', alpha=0.6, s=20)
       plt.xlabel('Feature 1')
       plt.ylabel('Feature 2')
       plt.title('Probability Contours')
       plt.grid(True, alpha=0.3)
       
       # Plot 3: Sigmoid function
       plt.subplot(1, 3, 3)
       z = np.linspace(-5, 5, 100)
       sigmoid = 1 / (1 + np.exp(-z))
       plt.plot(z, sigmoid, 'b-', linewidth=2)
       plt.xlabel('z = w^T φ(x)')
       plt.ylabel('P(y=1)')
       plt.title('Sigmoid Function')
       plt.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

   # Plot the results
   plot_logistic_results(X, y, model, basis)

Model Evaluation
----------------

Let's evaluate the model performance using various metrics:

.. code-block:: python

   from sklearn.metrics import classification_report, confusion_matrix
   import seaborn as sns

   # Compute predictions
   y_pred_proba, _ = model.predict(X)
   y_pred = (y_pred_proba > 0.5).astype(int)
   
   # Print classification report
   print("Classification Report:")
   print(classification_report(y, y_pred))
   
   # Plot confusion matrix
   plt.figure(figsize=(8, 6))
   cm = confusion_matrix(y, y_pred)
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title('Confusion Matrix')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.show()

Gradient Descent Training
------------------------

Let's examine the gradient descent process:

.. code-block:: python

   # Create a new model for gradient descent demonstration
   model_gd = mlai.LR(X, y, basis)
   
   # Track objective values during training
   objectives = []
   iterations = 50
   
   for i in range(iterations):
       # Compute gradient
       gradient = model_gd.gradient()
       
       # Update weights (simple gradient descent)
       learning_rate = 0.1
       model_gd.w_star = model_gd.w_star - learning_rate * gradient.flatten()
       
       # Compute objective
       objective = model_gd.objective()
       objectives.append(objective)
       
       if i % 10 == 0:
           print(f"Iteration {i}: Objective = {objective:.4f}")
   
   # Plot convergence
   plt.figure(figsize=(10, 6))
   plt.plot(objectives, 'b-', linewidth=2)
   plt.xlabel('Iteration')
   plt.ylabel('Negative Log-Likelihood')
   plt.title('Gradient Descent Convergence')
   plt.grid(True, alpha=0.3)
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
       basis = mlai.Basis(basis_func, num_basis, data_limits=[X.min(), X.max()])
       model = mlai.LR(X, y, basis)
       model.fit()
       
       # Evaluate
       y_pred_proba, _ = model.predict(X)
       y_pred = (y_pred_proba > 0.5).astype(int)
       accuracy = (y_pred == y).mean()
       
       results[name] = {
           'accuracy': accuracy,
           'model': model,
           'basis': basis
       }
       
       print(f"{name} basis: Accuracy = {accuracy:.3f}")
   
   # Plot decision boundaries for comparison
   fig, axes = plt.subplots(2, 2, figsize=(15, 12))
   axes = axes.ravel()
   
   for i, (name, result) in enumerate(results.items()):
       model = result['model']
       basis = result['basis']
       
       # Create grid
       x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
       y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
       xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                           np.linspace(y_min, y_max, 50))
       
       # Get predictions
       grid_points = np.c_[xx.ravel(), yy.ravel()]
       proba_grid, _ = model.predict(grid_points)
       proba_grid = proba_grid.reshape(xx.shape)
       
       # Plot
       axes[i].contourf(xx, yy, proba_grid, levels=20, cmap='RdYlBu', alpha=0.7)
       axes[i].scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', alpha=0.6, s=20)
       axes[i].scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', alpha=0.6, s=20)
       axes[i].contour(xx, yy, proba_grid, levels=[0.5], colors='black', linewidths=2)
       axes[i].set_title(f'{name} (Acc: {result["accuracy"]:.3f})')
       axes[i].set_xlabel('Feature 1')
       axes[i].set_ylabel('Feature 2')
       axes[i].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Key Concepts
------------

1. **Logistic Function**: The sigmoid function maps any real number to a probability between 0 and 1.

2. **Maximum Likelihood**: Logistic regression maximizes the likelihood of the observed data.

3. **Regularization**: Adding regularization terms helps prevent overfitting.

4. **Multiclass Extension**: Logistic regression can be extended to multiple classes using softmax.

Advantages and Limitations
-------------------------

**Advantages:**
- Simple and interpretable
- Provides probability estimates
- Works well with small datasets
- No assumptions about feature distributions

**Limitations:**
- Assumes linear relationship in log-odds
- May underperform on complex non-linear problems
- Sensitive to outliers
- Requires feature scaling for optimal performance

Further Reading
---------------

- `Logistic Regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ on Wikipedia
- `Generalized Linear Models <https://en.wikipedia.org/wiki/Generalized_linear_model>`_
- `Maximum Likelihood Estimation <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`_

This tutorial demonstrates how logistic regression provides a probabilistic framework for binary classification, forming the foundation for understanding more advanced classification techniques. 