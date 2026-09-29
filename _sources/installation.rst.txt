Installation Guide
==================

MLAI can be installed using either Poetry (recommended) or pip.

Prerequisites
------------

- Python 3.11 or higher
- pip or Poetry

Installation with Poetry
-----------------------

Poetry is the recommended way to install MLAI as it manages dependencies automatically:

.. code-block:: bash

   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Clone the repository
   git clone https://github.com/lawrennd/mlai.git
   cd mlai
   
   # Install dependencies
   poetry install

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

   # For mountain car demo
   pip install GPy
   
   # For GP tutorials
   pip install GPy

Verifying Installation
---------------------

To verify that MLAI is installed correctly:

.. code-block:: python

   import mlai
   print("MLAI version:", mlai.__version__)

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to get started with MLAI. 