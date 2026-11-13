Installation Guide
==================

MLAI can be installed with uv (recommended for contributors) or pip.

Prerequisites
------------

- Python 3.11 or higher
- `uv <https://docs.astral.sh/uv/>`_ or pip

Installation with uv
--------------------

uv is the recommended way to install MLAI for development. It creates a locked
virtual environment from ``uv.lock``:

.. code-block:: bash

   # Install uv if you do not have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone the repository
   git clone https://github.com/lawrennd/mlai.git
   cd mlai

   # Install runtime and development dependencies
   uv sync

Installation with pip
--------------------

You can also install MLAI using pip:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/lawrennd/mlai.git
   cd mlai

   # Install in development mode
   pip install -e .

Optional Dependencies
--------------------

For additional functionality, you can install optional dependencies:

.. code-block:: bash

   # For mountain car demo or GP tutorials
   pip install GPy

   # For graphical-model diagrams
   pip install daft-pgm

   # Or with uv extras
   uv sync --extra mountain-car
   uv sync --extra gp-tutorials
   uv sync --extra graphical-models

Verifying Installation
---------------------

To verify that MLAI is installed correctly:

.. code-block:: python

   import mlai
   print("MLAI version:", mlai.__version__)

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to get started with MLAI.
