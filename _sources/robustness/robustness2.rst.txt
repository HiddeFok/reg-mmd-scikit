Regression Robustness 
=====================

The robustness also translates to the regression procedure

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> from regmmd import MMDRegressor
   >>> from regmmd.utils import print_summary
   >>> from sklearn.linear_model import LinearRegression
   >>>
   >>> p = 3
   >>> x = rng.normal(loc=0, scale=1.5, size=(50, p))
   >>> beta = np.array([1, 2, 3])
   >>> noise = rng.normal(loc=0, scale=0.5, size=(50,))
   >>> y = x @ beta + noise
   >>> # We contaminate  only one sample
   >>> y[42] = 100
   >>>
   >>> # Fit a standard model
   >>> lin_reg = LinearRegression(fit_intercept=False)
   >>> _ = lin_reg.fit(x, y)
   >>>
   >>> mmd_reg = MMDRegressor(
   ...     model="linear-gaussian-loc",
   ...     par_v=None,
   ...     par_c=None,
   ...     fit_intercept=False,
   ...     bandwidth_X=1,
   ...     bandwidth_y=0.5,
   ...     kernel_X="Laplace",
   ...     kernel_y="Laplace",
   ...     solver={"burnin": 500,
   ...             "n_step": 10000,
   ...             "stepsize": 1, 
   ...             "epsilon": 1e-8
   ...      },
   ...      random_state=42
   ... )
   >>> res = mmd_reg.fit(X=x, y=y)  
   >>> lin_reg_error = np.abs(lin_reg.coef_ - beta).sum()
   >>> print(lin_reg.coef_)
   [-0.46385604  2.60706584  2.26142131]
   >>> print(lin_reg_error)
   2.8095005652996297
   >>> mmd_reg_error = np.abs(res["estimator"] - beta).sum()
   >>> print(res["estimator"])
   [0.74946784 3.5982815  2.87334419]
   >>> print(mmd_reg_error)
   1.9754694658571772
