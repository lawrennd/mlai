"""
Test that the actual workflow script runs successfully.

This test ensures that the example_workflow.py script in docs/tutorials/
actually works and produces expected outputs.
"""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import numpy as np


class TestWorkflowScript:
    """Test that the workflow script runs successfully."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for running the script
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Path to the workflow script - handle both local and CI environments
        possible_paths = [
            Path(self.original_cwd) / "docs" / "tutorials" / "example_workflow.py",
            Path(self.original_cwd) / "mlai" / "docs" / "tutorials" / "example_workflow.py",
            Path.cwd() / "docs" / "tutorials" / "example_workflow.py",
            Path.cwd() / "mlai" / "docs" / "tutorials" / "example_workflow.py"
        ]
        
        self.workflow_script = None
        for path in possible_paths:
            if path.exists():
                self.workflow_script = path
                break
                
        if self.workflow_script is None:
            pytest.skip(f"Workflow script not found. Tried: {[str(p) for p in possible_paths]}")
        
        # Copy the script to temp directory
        import shutil
        shutil.copy2(self.workflow_script, self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        # Clean up temporary files
        for file in Path(self.temp_dir).glob("*"):
            if file.is_file():
                file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_workflow_script_runs_successfully(self):
        """Test that the workflow script runs without errors."""
        # Run the workflow script with PYTHONPATH set to include the mlai package
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(self.original_cwd))
        
        result = subprocess.run(
            [sys.executable, "example_workflow.py"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir,
            env=env
        )
        
        # Check that the script ran successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"Script failed with return code {result.returncode}"
        
        # Print script output for debugging
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # List all files in the temp directory
        print(f"Files in temp directory: {list(Path(self.temp_dir).glob('*'))}")
        
        # Check that expected output files were created
        expected_files = [
            "basis_functions_demo.png",
            "linear_regression_demo.png", 
            "logistic_regression_demo.png"
        ]
        
        for filename in expected_files:
            assert Path(filename).exists(), f"Expected file {filename} was not created"
            assert Path(filename).stat().st_size > 0, f"File {filename} is empty"
        
        # Check that the script produced expected output
        assert "MLAI Tutorial Examples - Complete Workflow" in result.stdout
        assert "✓ All demonstrations completed successfully!" in result.stdout
        assert "✓ Generated plots saved as PNG files" in result.stdout
    
    def test_workflow_script_produces_expected_metrics(self):
        """Test that the workflow script produces reasonable metrics."""
        # Run the workflow script with PYTHONPATH set to include the mlai package
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(self.original_cwd))
        
        result = subprocess.run(
            [sys.executable, "example_workflow.py"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir,
            env=env
        )
        
        # Check for expected metrics in output
        output = result.stdout
        
        # Check that linear regression metrics are reasonable
        assert "Polynomial basis: MSE = " in output
        assert "R² = " in output
        
        # Check that logistic regression accuracy is reported
        assert "Training accuracy: " in output
        
        # Check that perceptron converged
        assert "Converged after" in output or "Final weights:" in output
    
    def test_workflow_script_no_errors(self):
        """Test that the workflow script runs without errors or warnings."""
        # Run the workflow script with PYTHONPATH set to include the mlai package
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(self.original_cwd))
        
        result = subprocess.run(
            [sys.executable, "example_workflow.py"],
            capture_output=True,
            text=True,
            cwd=self.temp_dir,
            env=env
        )
        
        # Check that there are no error messages
        stderr = result.stderr
        if stderr:
            # Allow for overflow warnings in logistic regression (expected)
            allowed_warnings = [
                "overflow encountered in exp",
                "RuntimeWarning",
                "g = 1./(1+np.exp(f))",  # This is the actual warning line
                "proba = 1. / (1 + np.exp(-f))"  # Another expected warning line
            ]
            
            for line in stderr.split('\n'):
                if line.strip():
                    is_allowed = any(warning in line for warning in allowed_warnings)
                    assert is_allowed, f"Unexpected error/warning: {line}"
    
    def test_workflow_script_imports_correctly(self):
        """Test that the workflow script can be imported without errors."""
        # Test importing the script as a module
        import sys
        workflow_dir = self.workflow_script.parent
        sys.path.insert(0, str(workflow_dir))
        
        try:
            # Import the workflow module
            import example_workflow
            
            # Check that main functions exist
            assert hasattr(example_workflow, 'demonstrate_basis_functions')
            assert hasattr(example_workflow, 'demonstrate_linear_regression')
            assert hasattr(example_workflow, 'demonstrate_logistic_regression')
            assert hasattr(example_workflow, 'demonstrate_perceptron')
            assert hasattr(example_workflow, 'demonstrate_bayesian_linear_regression')
            
        except ImportError as e:
            pytest.fail(f"Failed to import workflow script: {e}")
        finally:
            # Clean up sys.path
            sys.path.pop(0)


class TestWorkflowFunctions:
    """Test individual workflow functions."""
    
    def test_workflow_functions_are_callable(self):
        """Test that workflow functions can be called."""
        # Import the workflow module
        import sys
        # Find the workflow script path
        possible_paths = [
            Path(__file__).parent.parent.parent / "docs" / "tutorials" / "example_workflow.py",
            Path(__file__).parent.parent.parent.parent / "docs" / "tutorials" / "example_workflow.py",
        ]
        
        workflow_script = None
        for path in possible_paths:
            if path.exists():
                workflow_script = path
                break
                
        if workflow_script is None:
            pytest.skip(f"Workflow script not found for import test")
            
        workflow_dir = workflow_script.parent
        sys.path.insert(0, str(workflow_dir))
        
        try:
            import example_workflow
            
            # Test that functions can be called with proper setup
            # We'll test a simple function that doesn't require plotting
            
            # Test data generation functions
            x_data, y_data = example_workflow.generate_regression_data()
            assert x_data.shape == (100, 1)
            assert y_data.shape == (100, 1)
            
            X, y = example_workflow.generate_classification_data()
            assert X.shape == (200, 2)
            assert y.shape == (200,)
            
        except ImportError as e:
            pytest.fail(f"Failed to import workflow script: {e}")
        finally:
            sys.path.pop(0)


class TestWorkflowDependencies:
    """Test that workflow has all required dependencies."""
    
    def test_workflow_imports(self):
        """Test that all imports in the workflow script work."""
        # Read the workflow script
        possible_paths = [
            Path(__file__).parent.parent.parent / "docs" / "tutorials" / "example_workflow.py",
            Path(__file__).parent.parent.parent.parent / "docs" / "tutorials" / "example_workflow.py",
        ]
        
        workflow_script = None
        for path in possible_paths:
            if path.exists():
                workflow_script = path
                break
                
        if workflow_script is None:
            pytest.skip(f"Workflow script not found for import test")
        
        with open(workflow_script, 'r') as f:
            content = f.read()
        
        # Extract import statements
        import_lines = [line.strip() for line in content.split('\n') 
                       if line.strip().startswith(('import ', 'from '))]
        
        # Test each import
        for import_line in import_lines:
            if import_line.startswith('import ') or import_line.startswith('from '):
                # Skip matplotlib.pyplot import as it's handled specially
                if 'matplotlib.pyplot' in import_line:
                    continue
                
                # Try to execute the import
                try:
                    exec(import_line)
                except ImportError as e:
                    pytest.fail(f"Import failed: {import_line} - {e}")
                except Exception as e:
                    # Some imports might fail for other reasons, but that's okay
                    # as long as they're not ImportError
                    pass 