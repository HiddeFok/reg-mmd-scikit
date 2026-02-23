import numpy as np
import pytest
from regmmd.optimizer import _sgd_estimation, MMDResult

# def test_sgd_estimation():
#     X = np.random.rand(10, 2)
#     y = np.random.rand(10)
#     par_v = np.array([0.5, 0.5])
#     par_c = np.array([1.0])
#     result = _sgd_estimation(X, y, par_v, par_c, kernel="Gaussian")
#     assert isinstance(result, MMDResult)
#     assert "par_v_init" in result
#     assert "par_c_init" in result
#     assert "stepsize" in result
#     assert "estimator" in result

# def test_mmd_result_structure():
#     result = {
#         "par_v_init": np.array([0.5, 0.5]),
#         "par_c_init": np.array([1.0]),
#         "stepsize": 0.01,
#         "estimator": np.array([0.6, 0.6])
#     }
#     assert isinstance(result, dict)
#     assert all(key in result for key in ["par_v_init", "par_c_init", "stepsize", "estimator"])