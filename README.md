# MLAI: Machine Learning and Adaptive Intelligence Teaching Library

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

---

## 🚀 Installation

### With Poetry (recommended)
```bash
poetry install
```

### With pip
```bash
pip install -e .
```

---

## 🏁 Quick Start Example
```python
import mlai
# Example: Load a dataset, fit a model, and plot results
# (See tutorials and documentation for more details)
```

---

## 🗂️ Project Structure
```
mlai/                # Core Python package
├── mlai.py          # Main module
├── gp_tutorial.py   # Gaussian Process tutorial
├── deepgp_tutorial.py # Deep GP tutorial
├── mountain_car.py  # Mountain Car example
├── plot.py          # Plotting utilities
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

## 🧪 Testing
A comprehensive test framework is being developed (see [CIP-0002](cip/cip0002.md)).

- To run tests (when available):
  ```bash
  pytest
  ```
- Test coverage and CI/CD integration coming soon

---

## 📖 Documentation
- Sphinx documentation (coming soon)
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

