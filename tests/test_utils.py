import numpy as np
import pytest

from regmmd.utils import print_summary


def _base_result(trajectory):
    return {
        "par_v_init": np.array([0.0, 1.0]),
        "par_c_init": np.array([1.0]),
        "stepsize": 0.1,
        "estimator": np.array([0.5, 0.6]),
        "trajectory": trajectory,
        "bandwidth": 1.0,
        "convergence": 0,
    }


@pytest.mark.parametrize(
    "trajectory",
    [
        np.linspace(0.0, 1.0, 10),
        np.tile(np.linspace(0.0, 1.0, 10), (3, 1)),
    ],
    ids=["1d", "2d"],
)
def test_print_summary_runs(trajectory, capsys):
    res = _base_result(trajectory)
    print_summary(res)
    out = capsys.readouterr().out
    assert "MMD Result Summary Report" in out
    assert "End of Report" in out
    assert "Stepsize" in out
    assert "Estimated Parameters" in out


def test_print_summary_missing_optional_keys(capsys):
    res = {
        "par_v_init": np.array([0.0]),
        "stepsize": 0.5,
        "estimator": np.array([0.1]),
        "trajectory": np.array([0.0, 0.1, 0.2]),
    }
    print_summary(res)
    out = capsys.readouterr().out
    assert "Not provided" in out
