# MLAI: Machine Learning and Adaptive Intelligence Teaching Library

[![Tests](https://github.com/lawrennd/mlai/workflows/Tests/badge.svg)](https://github.com/lawrennd/mlai/actions/workflows/tests.yml)
[![Lint](https://github.com/lawrennd/mlai/workflows/Lint/badge.svg)](https://github.com/lawrennd/mlai/actions/workflows/lint.yml)
[![Documentation](https://github.com/lawrennd/mlai/workflows/Build%20and%20Deploy%20Documentation/badge.svg)](https://github.com/lawrennd/mlai/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/lawrennd/mlai/branch/main/graph/badge.svg)](https://codecov.io/gh/lawrennd/mlai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

MLAI is a Python package of simple models, tutorials, and plotting routines designed for teaching and lecturing on machine learning fundamentals. The project prioritizes clarity, mathematical transparency, and educational value, making it ideal for students, educators, and anyone seeking to understand the core ideas of machine learning.

---

## 📚 Project Origins

MLAI was originally developed to support the "Machine Learning and Adaptive Intelligence" course at the University of Sheffield. The codebase dates back to a refactoring of the course material in April 2018 and has since evolved to serve as a resource for teaching, learning, and experimenting with foundational machine learning concepts.

---

## ✨ Key Features
- *Educational Focus*: Code and documentation designed for clarity and learning
- *Mathematical Transparency*: Explicit links between code and mathematical concepts
- *Simple Models*: Implementations of core ML algorithms for teaching
- *Plotting Utilities*: Tools for visualizing data and models
- *Tutorials*: Example notebooks and scripts for hands-on learning
- *Open Science*: Encourages sharing, reproducibility, and contributions
- *Modular Design*: Clean, organized codebase with logical module separation (see [CIP-0006](cip/cip0006.md))

---

## 🚀 Installation

### With uv (recommended for contributors)
```bash
uv sync
```

### With pip
```bash
pip install -e .
```

---

## 🏁 Quick Start Example
```python
import mlai

# Linear regression example
X = mlai.generate_cluster_data(n_points_per_cluster=20)
y = X[:, 0] + 0.1 * mlai.np.random.randn(X.shape[0])

# Fit linear model
model = mlai.LM(X, y, mlai.linear)
model.fit()

# Plot results
mlai.plot.regression_plot(model, X, y)
```

For more examples, see the [tutorials](docs/) and [documentation](https://inverseprobability.com/mlai/).

---

## 🗂️ Project Structure
```
mlai/                # Core Python package (modular structure)
├── models.py        # Base model classes (Model, ProbModel, MapModel, ProbMapModel)
├── linear_models.py # Linear models (LM, BLM, LR) and basis functions
├── gaussian_processes.py # GP classes and kernel functions
├── neural_networks.py # Neural network classes, activations, and loss functions
├── utils.py         # Utility functions (gradient checks, file I/O, optimization)
├── dimred.py        # Dimensionality reduction and clustering algorithms
├── data.py          # Data generation utilities
├── experimental.py  # Experimental features (dropout networks)
├── plot.py          # Plotting and visualization utilities
├── gp_tutorial.py   # Gaussian Process tutorial
├── deepgp_tutorial.py # Deep GP tutorial
├── mountain_car.py  # Mountain Car example
backlog/             # Project backlog and task tracking
cip/                 # Code Improvement Proposals (CIPs)
docs/                # Documentation (Sphinx)
tenets/              # Project guiding principles
scripts/             # Utility scripts
```

---

## 🤝 Contributing
We welcome contributions! Please:
- Review the [MLAI Project Tenets](tenets/vibesafe-mlai-tenets.md) for our guiding principles
- Check open [CIPs](cip/) and [backlog](backlog/) items
- Follow good Python practices and prioritize clarity, especially for mathematical code
- Submit pull requests with clear explanations and, where possible, tests

---

## 🧹 Lint
The CI hard-fail gate checks syntax and undefined names (see [CIP-000A](cip/cip000A.md)).

```bash
uv run ruff check mlai
```

A second pass reports complexity and line length and does not fail the job.

---

## 🧪 Testing
A comprehensive test framework has been implemented (see [CIP-0002](cip/cip0002.md)).

- To run tests:
  ```bash
  pytest
  ```
- To run tests with coverage:
  ```bash
  pytest --cov=mlai --cov-report=html
  ```
- Tests run automatically on GitHub Actions for all pull requests
- Coverage reports are uploaded to [Codecov](https://codecov.io/gh/lawrennd/mlai)

---

## 📖 Documentation
- [Sphinx documentation](https://inverseprobability.com/mlai/) (hosted on GitHub Pages)
- [CIPs](cip/) for major improvements and design decisions
- [Tenets](tenets/vibesafe-mlai-tenets.md) for project philosophy

---

## 📜 License
MLAI is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📢 Citation
If you use MLAI for teaching, research, or publication, please cite the project and acknowledge its origins.

---

## 👤 Author
*Neil D. Lawrence*

For questions or suggestions, please open an issue or pull request on GitHub.

