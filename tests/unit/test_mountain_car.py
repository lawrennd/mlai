"""Unit tests for mlai.mountain_car (GPy optional; display mocked)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mountain_car():
    """Import mountain_car once with a temporary GPy stub.

    The stub must not remain in ``sys.modules`` after import — otherwise
    later tests that ``import GPy`` (e.g. plot.multiple_optima) get a
    MagicMock instead of ImportError and crash.
    """
    sys.modules.pop("mlai.mountain_car", None)
    gpy_stub = MagicMock(name="GPy")
    with patch.dict(sys.modules, {"GPy": gpy_stub}):
        mc = importlib.import_module("mlai.mountain_car")
    yield mc
    # Keep the loaded module for the rest of the session; GPy stub is gone.


@pytest.mark.unit
def test_mountain_car_imports(mountain_car):
    assert mountain_car is not None


@pytest.mark.unit
def test_mountain_car_ma_alias_resolves_to_utils(mountain_car):
    import mlai.utils

    assert mountain_car.ma is mlai.utils


@pytest.mark.unit
def test_mountain_car_key_functions_exist(mountain_car):
    expected = [
        "make_multi_output_multi_fidelity_kernel",
        "simulation",
        "low_cost_simulation",
        "run_simulation",
        "run_emulation",
        "calculate_linear_control",
        "make_gp_inputs",
        "v_simulation",
        "animate_frames",
        "invert_frames",
        "save_frames",
        "evalute_model",
        "add_data_to_gp",
        "emu_sim_comparison",
        "create_deep_multi_fidelity_models",
    ]
    for name in expected:
        assert hasattr(mountain_car, name)
        assert callable(getattr(mountain_car, name))


@pytest.mark.unit
def test_simulation_updates_position_and_velocity(mountain_car):
    d_pos, d_vel = mountain_car.simulation(np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)


@pytest.mark.unit
def test_low_cost_simulation_returns_deltas(mountain_car):
    d_pos, d_vel = mountain_car.low_cost_simulation(np.array([-0.5, 0.01, -1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)
    assert d_pos == d_vel


@pytest.mark.unit
def test_simulation_clips_position_at_bounds(mountain_car):
    d_pos, d_vel = mountain_car.simulation(np.array([0.59, 0.06, 1.0]))
    assert np.isfinite(d_pos)
    assert np.isfinite(d_vel)


@pytest.mark.unit
def test_v_simulation_returns_velocity_delta(mountain_car):
    out = mountain_car.v_simulation(np.array([0.0, 0.0, 1.0]))
    assert isinstance(out, float)
    assert np.isfinite(out)


@pytest.mark.unit
def test_calculate_linear_control_clips(mountain_car):
    gains = np.array([[10.0, 10.0, 10.0]])
    u = mountain_car.calculate_linear_control(np.array([1.0, 1.0]), gains)
    assert u.shape == (1,)
    assert u[0] == 1.0  # clipped

    gains_small = np.array([[0.1, 0.0, 0.0]])
    u2 = mountain_car.calculate_linear_control(np.array([0.5, 0.0]), gains_small)
    assert -1.0 <= u2[0] <= 1.0


@pytest.mark.unit
def test_make_gp_inputs_shapes(mountain_car):
    traj = np.array([[0.0, 0.0], [0.1, 0.01], [0.2, 0.02]])
    controls = np.array([[0.5], [-0.5]])
    X, y = mountain_car.make_gp_inputs(controls, traj)
    assert X.shape == (2, 3)
    assert y.shape == (2, 2)
    np.testing.assert_allclose(y[0], [0.1, 0.01])


@pytest.mark.unit
def test_add_data_to_gp_concatenates(mountain_car):
    gp = MagicMock()
    gp.X = np.array([[0.0, 0.0, 0.0]])
    gp.Y = np.array([[0.1]])
    new_x = np.array([[1.0, 0.0, 1.0]])
    new_y = np.array([[0.2]])
    out = mountain_car.add_data_to_gp(gp, new_x, new_y)
    assert out is gp
    args = gp.set_XY.call_args[0]
    assert args[0].shape == (2, 3)
    assert args[1].shape == (2, 1)


@pytest.mark.unit
def test_evalute_model_fidelity_branches(mountain_car):
    model = MagicMock()
    model.predict.return_value = (np.array([[0.05]]), np.array([[0.01]]))
    x = np.array([[0.0, 0.0, 1.0]])

    y_single = mountain_car.evalute_model(model, x, "single")
    assert y_single.shape[0] == 1
    model.predict.assert_called()

    y_multi = mountain_car.evalute_model(model, x, "multi-linear")
    call_x = model.predict.call_args[0][0]
    assert call_x.shape == (1, 4)
    assert call_x[0, -1] == 1.0
    assert y_multi is not None

    deep_models = [MagicMock(), MagicMock()]
    deep_models[0].predict.return_value = (np.array([[0.1]]), None)
    deep_models[1].predict.return_value = (np.array([[0.2]]), None)
    y_deep = mountain_car.evalute_model(deep_models, x, "multi-deep")
    assert np.allclose(y_deep, [[0.2]])


@pytest.mark.unit
def test_run_simulation_stops_when_done(mountain_car):
    env = MagicMock()
    env.reset.return_value = np.array([0.0, 0.0])
    env.action_space.shape = (1,)
    env.step.side_effect = [
        (np.array([0.1, 0.01]), 0.0, False, {}),
        (np.array([0.5, 0.02]), 1.0, True, {}),
    ]
    gains = np.array([[0.0, 0.0, 0.0]])

    cost, traj, controls, frames = mountain_car.run_simulation(
        env, gains, render=False
    )
    assert traj.shape[0] >= 2
    assert controls.shape[1] == 1
    assert frames == []
    assert isinstance(cost, (int, float, np.floating))


@pytest.mark.unit
def test_run_simulation_hits_horizon_without_done(mountain_car):
    env = MagicMock()
    env.reset.return_value = np.array([0.0, 0.0])
    env.action_space.shape = (1,)
    env.step.return_value = (np.array([0.1, 0.0]), 0.0, False, {})
    gains = np.array([[0.0, 0.0, 0.0]])
    with patch.object(mountain_car, "N_STEPS_MAX", 2):
        cost, traj, controls, frames = mountain_car.run_simulation(
            env, gains, render=False
        )
    assert traj.shape[0] == 2
    assert controls.shape[0] == 2
    assert frames == []
    assert cost >= 0


@pytest.mark.unit
def test_run_simulation_with_render_collects_frames(mountain_car):
    env = MagicMock()
    env.reset.return_value = np.array([0.0, 0.0])
    env.action_space.shape = (1,)
    env.render.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    env.step.return_value = (np.array([0.5, 0.0]), 1.0, True, {})
    gains = np.array([[0.0, 0.0, 0.0]])

    _, _, _, frames = mountain_car.run_simulation(env, gains, render=True)
    assert len(frames) == 1
    env.render.assert_called_with(mode="rgb_array")


@pytest.mark.unit
def test_run_emulation_success_and_horizon(mountain_car):
    # run_emulation does ``observation[i] += next_state_mean``; mean must be scalar-like.
    dynamics = [MagicMock(), MagicMock()]
    dynamics[0].predict.return_value = (np.float64(0.05), None)
    dynamics[1].predict.return_value = (np.float64(0.01), None)
    gains = np.array([[0.0, 0.0, 0.0]])
    x0 = np.array([0.4, 0.0])

    cost, traj, controls = mountain_car.run_emulation(
        dynamics, gains, x0.copy(), fidelity="single"
    )
    assert traj.shape[0] >= 2
    assert controls.shape[1] == 1
    assert cost <= 0  # success bonus when position > 0.45

    dynamics_slow = [MagicMock(), MagicMock()]
    dynamics_slow[0].predict.return_value = (np.float64(1e-6), None)
    dynamics_slow[1].predict.return_value = (np.float64(0.0), None)
    with patch.object(mountain_car, "N_STEPS_MAX", 3):
        cost2, traj2, _ = mountain_car.run_emulation(
            dynamics_slow, gains, np.array([0.0, 0.0]), fidelity="single"
        )
    assert traj2.shape[0] == 3
    assert cost2 > 0


@pytest.mark.unit
def test_make_multi_output_multi_fidelity_kernel(mountain_car):
    kern = mountain_car.make_multi_output_multi_fidelity_kernel(3)
    assert kern is not None


@pytest.mark.unit
def test_create_deep_multi_fidelity_models(mountain_car):
    gpy = mountain_car.GPy
    m1 = MagicMock()
    m2 = MagicMock()
    m1.predict.return_value = (np.zeros((4, 1)), None)
    gpy.models.GPRegression.side_effect = [m1, m2]
    gpy.kern.RBF.return_value = MagicMock()

    x = np.random.randn(4, 3)
    y1 = np.random.randn(4, 1)
    y2 = np.random.randn(4, 1)
    out1, out2 = mountain_car.create_deep_multi_fidelity_models(x, y1, y2)
    assert out1 is m1
    assert out2 is m2
    m1.optimize_restarts.assert_called()
    m2.optimize_restarts.assert_called()
    gpy.models.GPRegression.side_effect = None
    gpy.models.GPRegression.reset_mock()


@pytest.mark.unit
def test_plot_control_slices(mountain_car):
    emulator = MagicMock()
    emulator.predict.return_value = (np.zeros((50 * 50, 1)), None)
    pc = mountain_car.plot_control(emulator, fidelity="single")
    pc.plot_slices(0.0)
    mountain_car.plt.close("all")


@pytest.mark.unit
def test_invert_and_animate_and_save_frames(mountain_car, tmp_path):
    frames = [
        np.full((8, 8, 3), 255, dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    inverted = mountain_car.invert_frames(frames)
    assert len(inverted) == 2
    assert inverted[0].shape == (8, 8, 4)

    anim = mountain_car.animate_frames(frames)  # title is None → full-bleed axes
    assert anim is not None
    anim_titled = mountain_car.animate_frames(frames, title="demo")
    assert anim_titled is not None
    mountain_car.plt.close("all")

    with patch.object(mountain_car.ma, "write_animation_html") as write_html:
        mountain_car.save_frames(
            frames, "car.html", diagrams=str(tmp_path), inverted=True
        )
        write_html.assert_called_once()


@pytest.mark.unit
def test_emu_sim_comparison(mountain_car, tmp_path):
    env = MagicMock()
    env.reset.return_value = np.array([0.0, 0.0])
    env.action_space.shape = (1,)
    env.step.return_value = (np.array([0.5, 0.0]), 1.0, True, {})

    dynamics = [MagicMock(), MagicMock()]
    dynamics[0].predict.return_value = (np.float64(0.05), None)
    dynamics[1].predict.return_value = (np.float64(0.0), None)
    gains = np.array([[0.0, 0.0, 0.0]])

    with (
        patch.object(mountain_car.plt, "show"),
        patch.object(mountain_car.ma, "write_figure") as write_fig,
    ):
        mountain_car.emu_sim_comparison(
            env, gains, dynamics, fidelity="single", diagrams=str(tmp_path)
        )
        write_fig.assert_called_once()
    mountain_car.plt.close("all")
