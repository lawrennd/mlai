Quick Start Guide
================

This guide will help you get started with MLAI quickly. We'll cover the basic concepts and show you how to run your first examples.

Basic Usage
----------

MLAI provides simple, educational implementations of machine learning algorithms. Here's a basic example:

.. code-block:: python

   import mlai
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Generate some sample data
   X = np.linspace(0, 10, 100).reshape(-1, 1)
   y = np.sin(X).flatten() + 0.1 * np.random.randn(100)
   
   # Create and fit a simple model
   model = mlai.GaussianProcess(X, y)
   model.fit()
   
   # Make predictions
   X_test = np.linspace(0, 10, 200).reshape(-1, 1)
   y_pred, y_var = model.predict(X_test)
   
   # Plot results
   plt.figure(figsize=(10, 6))
   plt.scatter(X, y, alpha=0.5, label='Data')
   plt.plot(X_test, y_pred, 'r-', label='Prediction')
   plt.fill_between(X_test.flatten(), 
                   y_pred - 2*np.sqrt(y_var), 
                   y_pred + 2*np.sqrt(y_var), 
                   alpha=0.3, label='95% Confidence')
   plt.legend()
   plt.show()

Tutorials
---------

MLAI includes several tutorials to help you learn machine learning concepts:

- **Gaussian Process Tutorial** (:doc:`tutorials/gp_tutorial`): Learn about Gaussian Processes
- **Deep GP Tutorial** (:doc:`tutorials/deepgp_tutorial`): Explore Deep Gaussian Processes
- **Mountain Car Example** (:doc:`tutorials/mountain_car`): Reinforcement learning example

Plotting Utilities
-----------------

MLAI provides convenient plotting utilities for machine learning visualizations:

.. code-block:: python

   import mlai.plot as ma_plot
   
   # Use MLAI's plotting utilities
   ma_plot.set_defaults()  # Set default plotting parameters
   
   # Create publication-quality plots
   fig, ax = ma_plot.new_xy_figure()
   # ... your plotting code here

Key Concepts
-----------

MLAI is designed with these principles in mind:

1. **Clarity**: Code is written to be easily understood
2. **Mathematical Transparency**: Mathematical concepts are explicit in the code
3. **Educational Focus**: Every function serves a pedagogical purpose
4. **Reproducibility**: Examples can be run end-to-end

Next Steps
----------

- Explore the :doc:`api/index` for detailed API documentation
- Check out the :doc:`tutorials/index` for hands-on examples
- Read about our :doc:`tenets` to understand the project philosophy 