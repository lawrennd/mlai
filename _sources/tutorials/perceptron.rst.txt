Perceptron Algorithm Tutorial
=============================

The perceptron is one of the earliest and simplest machine learning algorithms for binary classification. This tutorial will walk you through the mathematical foundations, implementation, and practical usage of the perceptron algorithm using MLAI.

Mathematical Background
----------------------

The perceptron algorithm is a linear classifier that learns to separate two classes of data points using a hyperplane. Given input data :math:`\mathbf{x} \in \mathbb{R}^d` and binary labels :math:`y \in \{-1, +1\}`, the perceptron learns a weight vector :math:`\mathbf{w} \in \mathbb{R}^d` and bias :math:`b \in \mathbb{R}` such that:

.. math::

   f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)

The algorithm works by iteratively updating the weights when it makes a mistake:

.. math::

   \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \alpha \cdot y_i \cdot \mathbf{x}_i

where :math:`\alpha` is the learning rate.

Implementation in MLAI
---------------------

MLAI provides a simple implementation of the perceptron algorithm. Let's explore how to use it:

.. code-block:: python

   import numpy as np
   import mlai.mlai as mlai
   import matplotlib.pyplot as plt

   # Generate some linearly separable data
   np.random.seed(42)
   n_samples = 100
   n_features = 2
   
   # Create two classes
   X_plus = np.random.randn(n_samples//2, n_features) + np.array([2, 2])
   X_minus = np.random.randn(n_samples//2, n_features) + np.array([-2, -2])
   
   # Combine the data
   X = np.vstack([X_plus, X_minus])
   y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])

   # Initialize perceptron weights
   w, b = mlai.init_perceptron(X_plus, X_minus, seed=42)
   print(f"Initial weights: {w}")
   print(f"Initial bias: {b}")

Training the Perceptron
----------------------

Now let's train the perceptron on our data:

.. code-block:: python

   # Training parameters
   max_iterations = 100
   learning_rate = 0.1
   
   # Training loop
   for iteration in range(max_iterations):
       # Update weights for one epoch
       w, b, x_selected, updated = mlai.update_perceptron(w, b, X_plus, X_minus, learning_rate)
       
       if not updated:
           print(f"Converged after {iteration} iterations")
           break
   
   print(f"Final weights: {w}")
   print(f"Final bias: {b}")

Visualizing the Results
----------------------

Instead of manually plotting the decision boundary, you can use MLAI's built-in perceptron visualization tools for a more informative plot. These tools show the decision boundary, the weight vector, and histograms of the projections for each class.

.. code-block:: python

   import mlai.plot as plot
   import matplotlib.pyplot as plt

   # Prepare the figure and axes
   fig, axes = plt.subplots(1, 2, figsize=(12, 6))

   # Initialize the perceptron plot
   handles = plot.init_perceptron(fig, axes, X_plus, X_minus, w, b, fontsize=16)
   plt.show()

   # During training, you can update the plot after each weight update:
   for iteration in range(max_iterations):
       w, b, x_selected, updated = mlai.update_perceptron(w, b, X_plus, X_minus, learning_rate)
       handles = plot.update_perceptron(handles, fig, axes, X_plus, X_minus, iteration, w, b)
       plt.pause(0.1)  # Pause to visualize the update
       if not updated:
           print(f"Converged after {iteration} iterations")
           break
   plt.show()

This approach provides a dynamic and educational visualization of how the perceptron algorithm updates its decision boundary and weight vector during training.

Key Concepts
------------

1. **Linear Separability**: The perceptron only works when the data is linearly separable. If the classes cannot be separated by a hyperplane, the algorithm will not converge.

2. **Convergence**: The perceptron convergence theorem states that if the data is linearly separable, the perceptron will converge in a finite number of steps.

3. **Learning Rate**: The learning rate controls how much the weights are updated on each mistake. A larger learning rate leads to faster convergence but may cause instability.

4. **Bias Term**: The bias term allows the decision boundary to not pass through the origin, making the model more flexible.

Limitations
-----------

- Only works for linearly separable data
- Sensitive to the order of training examples
- May not find the optimal separating hyperplane
- Binary classification only

Extensions
----------

The perceptron algorithm has inspired many modern machine learning techniques:

- **Support Vector Machines (SVMs)**: Find the optimal separating hyperplane
- **Neural Networks**: Multi-layer perceptrons for complex decision boundaries
- **Online Learning**: Update weights incrementally as new data arrives

Further Reading
---------------

- `Perceptron Algorithm <https://en.wikipedia.org/wiki/Perceptron>`_ on Wikipedia
- `Rosenblatt's Original Paper <https://psycnet.apa.org/record/1958-00856-001>`_
- `Perceptron Convergence Theorem <https://en.wikipedia.org/wiki/Perceptron_convergence_theorem>`_

This tutorial demonstrates the fundamental concepts of linear classification and provides a foundation for understanding more advanced machine learning algorithms. 