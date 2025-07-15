import pytest
import mlai


@pytest.mark.unit
def test_mlai_module_import():
    """Test that the mlai module can be imported."""
    assert hasattr(mlai, 'mlai')


@pytest.mark.unit
def test_gpy_availability_flag_exists():
    """Test that GPY_AVAILABLE flag exists."""
    assert hasattr(mlai, 'GPY_AVAILABLE')
    assert isinstance(mlai.GPY_AVAILABLE, bool)


@pytest.mark.unit
def test_key_functions_available_at_package_level():
    """Test that key functions are available at the package level."""
    expected_functions = [
        'write_figure',
        'write_animation', 
        'Basis',
        'linear',
        'icm_cov',
        'slfm_cov'
    ]
    
    for func_name in expected_functions:
        assert hasattr(mlai, func_name), f"Function {func_name} not found at package level"


@pytest.mark.unit
def test_key_functions_are_callable():
    """Test that key functions are callable (not just attributes)."""
    expected_functions = [
        'write_figure',
        'write_animation',
        'Basis',
        'linear', 
        'icm_cov',
        'slfm_cov'
    ]
    
    for func_name in expected_functions:
        func = getattr(mlai, func_name)
        assert callable(func), f"Function {func_name} is not callable"


@pytest.mark.unit
def test_package_has_expected_attributes():
    """Test that the package has all expected attributes."""
    expected_attributes = [
        'mlai',           # Core module
        'GPY_AVAILABLE',  # GPy availability flag
        'write_figure',   # Key functions
        'write_animation',
        'Basis',
        'linear',
        'icm_cov', 
        'slfm_cov'
    ]
    
    for attr in expected_attributes:
        assert hasattr(mlai, attr), f"Attribute {attr} not found in mlai package"


@pytest.mark.unit
def test_gpy_dependent_modules_conditionally_available():
    """Test that GPy-dependent modules are conditionally available based on GPY_AVAILABLE."""
    # Import mlai fresh to avoid test isolation issues
    import importlib
    import sys
    
    # Remove mlai from sys.modules to force a fresh import
    if 'mlai' in sys.modules:
        del sys.modules['mlai']
    
    # Import mlai fresh
    import mlai
    
    if mlai.GPY_AVAILABLE:
        # If GPy is available, these modules should be imported
        assert hasattr(mlai, 'mountain_car'), "mountain_car should be available when GPy is available"
        assert hasattr(mlai, 'gp_tutorial'), "gp_tutorial should be available when GPy is available"
        assert hasattr(mlai, 'deepgp_tutorial'), "deepgp_tutorial should be available when GPy is available"
    else:
        # If GPy is not available, these modules should not be imported
        assert not hasattr(mlai, 'mountain_car'), "mountain_car should not be available when GPy is unavailable"
        assert not hasattr(mlai, 'gp_tutorial'), "gp_tutorial should not be available when GPy is unavailable"
        assert not hasattr(mlai, 'deepgp_tutorial'), "deepgp_tutorial should not be available when GPy is unavailable" 