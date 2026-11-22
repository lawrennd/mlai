Contributing to MLAI
===================

Thank you for your interest in contributing to MLAI! This guide will help you get started.

Project Philosophy
-----------------

MLAI follows the principles outlined in our :doc:`tenets`. Before contributing, please familiarize yourself with these guiding principles:

1. *Clarity Over Cleverness*: Code should be easy to understand
2. *Mathematical Transparency*: Mathematical concepts should be explicit
3. *Educational Focus*: Every contribution should serve a pedagogical purpose
4. *Good Python Practices*: Follow PEP 8 and modern Python conventions
5. *Reproducibility*: Examples should be runnable end-to-end
6. *Inclusivity*: Documentation should be accessible to diverse audiences
7. *Open Science*: Encourage sharing and collaboration

Getting Started
--------------

1. *Fork the repository* on GitHub
2. *Clone your fork* locally
3. *Create a virtual environment*:
   .. code-block:: bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
4. *Install dependencies*:
   .. code-block:: bash
      pip install -e .
      pip install -r requirements-dev.txt  # If available

Development Workflow
-------------------

1. *Check existing issues and CIPs*: Look at the :doc:`cip/index` and :doc:`backlog/index`
2. *Create a feature branch*: `git checkout -b feature/your-feature-name`
3. *Make your changes*: Follow the coding standards below
4. *Test your changes*: Run the test suite (when available)
5. *Update documentation*: Add or update docstrings and documentation
6. *Submit a pull request*: Include a clear description of your changes

Coding Standards
----------------

- *Style*: Follow PEP 8 and PEP 257
- *Type hints*: Use type hints where they improve clarity
- *Docstrings*: Use NumPy or Google docstring format
- *Mathematical notation*: Include mathematical equations in docstrings
- *Examples*: Provide working examples in docstrings

Example docstring:

.. code-block:: python

   def gaussian_kernel(x1, x2, lengthscale=1.0):
       """
       Compute the Gaussian (RBF) kernel between two points.
       
       The Gaussian kernel is defined as:
       
       .. math::
          k(x_1, x_2) = \exp\left(-\frac{\|x_1 - x_2\|^2}{2\ell^2}\right)
       
       where :math:`\ell` is the lengthscale parameter.
       
       Parameters
       ----------
       x1 : array-like
           First input point
       x2 : array-like
           Second input point
       lengthscale : float, default=1.0
           Lengthscale parameter :math:`\ell`
       
       Returns
       -------
       float
           Kernel value
       
       Examples
       --------
       >>> gaussian_kernel([0], [1], lengthscale=1.0)
       0.6065306597126334
       """
       pass

Testing
-------

When the test framework is implemented (see :doc:`cip/cip0002`), please:

- Write tests for new functionality
- Ensure all tests pass
- Maintain or improve test coverage
- Include property-based tests for mathematical functions

Documentation
------------

- Update docstrings for any functions you modify
- Add examples to the tutorials if relevant
- Update this contributing guide if needed
- Follow the documentation style guide

Submitting Changes
-----------------

1. *Write a clear commit message* following conventional commits
2. *Reference related issues or CIPs* in your commit message
3. *Provide a detailed pull request description*
4. *Include examples* of how your changes work
5. *Link to relevant documentation* or discussions

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

Questions?
----------

If you have questions about contributing:

- Check the :doc:`tenets` for project philosophy
- Review existing :doc:`cip/index` for design decisions
- Open an issue on GitHub for specific questions
- Join our community discussions

Thank you for contributing to MLAI! 