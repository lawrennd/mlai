#!/usr/bin/env python3
"""
MLAI Tutorial Examples - Complete Workflow

This script demonstrates the key concepts from all tutorials working together:
1. Basis Functions
2. Linear Regression
3. Logistic Regression
4. Perceptron Algorithm

Run this script to see all concepts in action with visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mlai.mlai as mlai

def setup_plotting():
    """Set up matplotlib for better plots."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def generate_regression_data():
    """Generate synthetic regression data."""
    np.random.seed(42)
    n_samples = 100
    
    # Create non-linear data with noise
    x_data = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_data = 2 * x_data**2 - 3 * x_data + 1 + 0.5 * np.random.randn(n_samples, 1)
    
    return x_data, y_data

def generate_classification_data():
    """Generate synthetic classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 2
    
    # Create two classes with some overlap
    X_class0 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], n_samples//2)
    X_class1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n_samples//2)
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    return X, y

def demonstrate_basis_functions():
    """Demonstrate different basis functions."""
    print("=== Basis Functions Demonstration ===")
    
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    # Test different basis functions
    basis_types = [
        ('Linear', mlai.linear, 1),
        ('Polynomial', mlai.polynomial, 3),
        ('Radial', mlai.radial, 5),
        ('Fourier', mlai.fourier, 4)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, basis_func, num_basis) in enumerate(basis_types):
        basis = mlai.Basis(basis_func, num_basis, data_limits=[float(np.min(x)), float(np.max(x))])
        Phi = basis.Phi(x)
        
        for j in range(Phi.shape[1]):
            axes[i].plot(x, Phi[:, j], label=f'Basis {j+1}', linewidth=2)
        
        axes[i].set_title(f'{name} Basis Functions')
        axes[i].set_xlabel('Input x')
        axes[i].set_ylabel('Basis Function Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basis_functions_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Basis functions demonstration completed")

def demonstrate_linear_regression():
    """Demonstrate linear regression with different basis functions."""
    print("\n=== Linear Regression Demonstration ===")
    
    x_data, y_data = generate_regression_data()
    print(f"x_data shape: {x_data.shape}, type: {type(x_data)}")
    print(f"y_data shape: {y_data.shape}, type: {type(y_data)}")
    print(f"x_data min: {np.min(x_data)}, max: {np.max(x_data)}")
    
    # Test different basis functions
    basis_configs = [
        ('Linear', mlai.linear, 1),
        ('Polynomial', mlai.polynomial, 3),
        ('Radial', mlai.radial, 5),
        ('Fourier', mlai.fourier, 4)
    ]
    
    results = {}
    
    for name, basis_func, num_basis in basis_configs:
        print(f"\n--- Testing {name} basis ---")
        data_limits = [float(np.min(x_data)), float(np.max(x_data))]
        print(f"data_limits: {data_limits}, type: {type(data_limits)}")
        
        basis = mlai.Basis(basis_func, num_basis, data_limits=data_limits)
        print(f"Basis created successfully")
        
        try:
            model = mlai.LM(x_data, y_data, basis)
            print(f"Model created successfully")
            model.fit()
            print(f"Model fitted successfully")
            
            y_pred, _ = model.predict(x_data)
            mse = np.mean((y_data - y_pred) ** 2)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            results[name] = {
                'mse': mse,
                'r2': r2,
                'model': model,
                'basis': basis
            }
            
            print(f"{name} basis: MSE = {mse:.4f}, R² = {r2:.4f}")
        except Exception as e:
            print(f"Error with {name} basis: {e}")
            continue
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        model = result['model']
        
        # Plot data and fit
        axes[i].scatter(x_data, y_data, c='blue', alpha=0.6, s=20)
        
        x_smooth = np.linspace(np.min(x_data), np.max(x_data), 200).reshape(-1, 1)
        y_smooth, _ = model.predict(x_smooth)
        axes[i].plot(x_smooth, y_smooth, 'r-', linewidth=2)
        
        axes[i].set_title(f'{name} (MSE: {result["mse"]:.4f}, R²: {result["r2"]:.3f})')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Linear regression demonstration completed")

def demonstrate_logistic_regression():
    """Demonstrate logistic regression for classification."""
    print("\n=== Logistic Regression Demonstration ===")
    
    X, y = generate_classification_data()
    
    # Ensure y is 2D for the model
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Use the new multivariate RBF basis with a fixed random seed
    basis = mlai.Basis(lambda x, num_basis: mlai.radial_multivariate(x, num_basis=num_basis, random_state=42), 10)
    
    # Create and fit logistic regression model
    model = mlai.LR(X, y, basis)
    model.fit()
    
    # Make predictions
    y_pred_proba, Phi = model.predict(X)
    if y_pred_proba.ndim == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate accuracy
    accuracy = (y_pred == y).mean()
    print(f"Training accuracy: {accuracy:.3f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Create grid for visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions for grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    proba_grid, _ = model.predict(grid_points)
    if proba_grid.ndim == 1:
        proba_grid = proba_grid.reshape(-1, 1)
    proba_grid = proba_grid.reshape(xx.shape)
    
    # Plot 1: Data points and decision boundary
    plt.subplot(1, 3, 1)
    plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], c='red', alpha=0.6, label='Class 0')
    plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], c='blue', alpha=0.6, label='Class 1')
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
    plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], c='red', alpha=0.6, s=20)
    plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], c='blue', alpha=0.6, s=20)
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
    plt.savefig('logistic_regression_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Logistic regression demonstration completed")

def demonstrate_perceptron():
    """Demonstrate the perceptron algorithm."""
    print("\n=== Perceptron Algorithm Demonstration ===")
    
    # Generate linearly separable data
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
    w, b, x_select = mlai.init_perceptron(X_plus, X_minus, seed=42)
    print(f"Initial weights: {w}")
    print(f"Initial bias: {b}")
    
    # Prepare the figure and axes for MLAI perceptron visualization
    import mlai.plot as plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    handles = plot.init_perceptron(fig, axes, X_plus, X_minus, w, b, fontsize=16)
    plt.show(block=False)
    
    # Training loop with dynamic visualization
    max_iterations = 100
    learning_rate = 0.1
    convergence_iteration = None
    
    for iteration in range(max_iterations):
        w, b, x_selected, updated = mlai.update_perceptron(w, b, X_plus, X_minus, learning_rate)
        handles = plot.update_perceptron(handles, fig, axes, X_plus, X_minus, iteration, w, b)
        plt.pause(0.1)
        if not updated:
            convergence_iteration = iteration
            print(f"Converged after {iteration} iterations")
            break
    
    if convergence_iteration is None:
        print("Did not converge within maximum iterations")
    
    print(f"Final weights: {w}")
    print(f"Final bias: {b}")
    plt.show()
    
    print("✓ Perceptron demonstration completed")

def demonstrate_bayesian_linear_regression():
    """Demonstrate Bayesian linear regression."""
    print("\n=== Bayesian Linear Regression Demonstration ===")
    
    x_data, y_data = generate_regression_data()
    
    # Create polynomial basis
    basis = mlai.Basis(mlai.polynomial, 3, data_limits=[float(np.min(x_data)), float(np.max(x_data))])
    
    # Create Bayesian linear model
    blm = mlai.BLM(x_data, y_data, basis)
    blm.fit()
    
    # Make predictions with uncertainty
    x_test = np.linspace(np.min(x_data), np.max(x_data), 100).reshape(-1, 1)
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
    
    plt.savefig('bayesian_linear_regression_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Bayesian linear regression demonstration completed")

def main():
    """Run all demonstrations."""
    print("MLAI Tutorial Examples - Complete Workflow")
    print("=" * 50)
    
    setup_plotting()
    
    try:
        # Run all demonstrations
        demonstrate_basis_functions()
        demonstrate_linear_regression()
        demonstrate_logistic_regression()
        demonstrate_perceptron()
        demonstrate_bayesian_linear_regression()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("✓ Generated plots saved as PNG files")
        print("\nKey concepts demonstrated:")
        print("- Basis function transformations")
        print("- Linear regression with different bases")
        print("- Logistic regression for classification")
        print("- Perceptron algorithm for binary classification")
        print("- Bayesian linear regression with uncertainty")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check that all required dependencies are installed.")

if __name__ == "__main__":
    main() 