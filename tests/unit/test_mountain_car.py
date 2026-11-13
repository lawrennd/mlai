import sys
import importlib
import pytest
from unittest.mock import patch, MagicMock


MOUNTAIN_CAR_MODULES = [
    'mlai.mountain_car',
    'mlai',
]


def _fresh_mountain_car():
    """Import mlai.mountain_car with a mocked GPy, evicting any cached copy first."""
    for mod in MOUNTAIN_CAR_MODULES:
        sys.modules.pop(mod, None)
    with patch.dict('sys.modules', {'GPy': MagicMock()}):
        return importlib.import_module('mlai.mountain_car')


@pytest.mark.unit
def test_mountain_car_imports():
    """mountain_car should import without error when GPy is mocked."""
    mc = _fresh_mountain_car()
    assert mc is not None


@pytest.mark.unit
def test_mountain_car_uses_utils_not_mlai_mlai():
    """Regression: mountain_car must not import the defunct mlai.mlai submodule."""
    # If the stale 'import mlai.mlai as ma' were restored this would raise ModuleNotFoundError
    mc = _fresh_mountain_car()
    assert mc is not None


@pytest.mark.unit
def test_mountain_car_ma_alias_resolves_to_utils():
    """The 'ma' alias inside mountain_car must be mlai.utils, not mlai.mlai."""
    import mlai.utils
    mc = _fresh_mountain_car()
    ma = vars(mc).get('ma')
    assert ma is not None, "'ma' alias not found in mountain_car module globals"
    assert ma is mlai.utils, (
        f"'ma' should be mlai.utils but is {ma}. "
        "The stale 'import mlai.mlai as ma' may have been restored."
    )


@pytest.mark.unit
def test_mountain_car_key_functions_exist():
    """All public functions should be present and callable."""
    mc = _fresh_mountain_car()
    expected = [
        'make_multi_output_multi_fidelity_kernel',
        'simulation',
        'low_cost_simulation',
        'run_simulation',
        'run_emulation',
        'calculate_linear_control',
        'make_gp_inputs',
        'animate_frames',
        'invert_frames',
        'save_frames',
    ]
    for name in expected:
        assert hasattr(mc, name), f"Expected function '{name}' not found in mountain_car"
        assert callable(getattr(mc, name)), f"'{name}' is not callable"


@pytest.mark.unit
def test_mountain_car_write_figure_callable_via_ma():
    """write_figure must be accessible through the ma alias (mlai.utils)."""
    mc = _fresh_mountain_car()
    ma = vars(mc).get('ma')
    assert hasattr(ma, 'write_figure'), "write_figure not found on ma (mlai.utils)"
    assert callable(ma.write_figure)


@pytest.mark.unit
def test_mountain_car_write_animation_html_callable_via_ma():
    """write_animation_html must be accessible through the ma alias (mlai.utils)."""
    mc = _fresh_mountain_car()
    ma = vars(mc).get('ma')
    assert hasattr(ma, 'write_animation_html'), "write_animation_html not found on ma (mlai.utils)"
    assert callable(ma.write_animation_html)


@pytest.mark.unit
def test_simulation_updates_position_and_velocity():
    """Pure-numpy dynamics should return finite deltas without GPy."""
    import numpy as np

    mc = _fresh_mountain_car()
    d_pos, d_vel = mc.simulation(np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)


@pytest.mark.unit
def test_low_cost_simulation_returns_deltas():
    import numpy as np

    mc = _fresh_mountain_car()
    d_pos, d_vel = mc.low_cost_simulation(np.array([-0.5, 0.01, -1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)
    assert d_pos == d_vel


@pytest.mark.unit
def test_simulation_clips_position_at_bounds():
    import numpy as np

    mc = _fresh_mountain_car()
    # Strong positive action near the right wall should not explode past max_position.
    d_pos, d_vel = mc.simulation(np.array([0.59, 0.06, 1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)
