Example 2: Regression
======================

Here is an example of how to use the ``regmmd.regression`` module. In the
following example the linear beta coefficient is estimated in a Gamma Regression 
task.

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> from regmmd import MMDRegressor
   >>> from regmmd.utils import print_summary
   >>> from regmmd.regression import GammaRegressionLoc
   >>> n = 1000
   >>> p = 4
   >>> beta = np.arange(1, 5)
   >>> model_true = GammaRegressionLoc(par_v=beta, par_c=1, random_state=12)
   >>> 
   >>> rng = np.random.default_rng(seed=123)
   >>> X = rng.normal(loc=0, scale=1, size=(n, p))
   >>> mu_given_x = model_true.predict(X=X)
   >>> y = model_true.sample_n(n=n, mu_given_x=mu_given_x)
   >>>
   >>> beta_init = np.array([0.5, 1.5, 2.5, 3.2])
   >>>
   >>> mmd_reg = MMDRegressor(
   ...     model="gamma-regression-loc",
   ...     par_v=beta_init,
   ...     par_c=1.5,
   ...     fit_intercept=False,
   ...     bandwidth_X=0,
   ...     bandwidth_y=1,
   ...     kernel_y="Gaussian",
   ...     solver={
   ...         "type": "SGD",
   ...         "burnin": 500,
   ...         "n_step": 10000,
   ...         "stepsize": 1,
   ...         "epsilon": 1e-8,
   ...     },
   ...     random_state=20
   ... )
   >>> res = mmd_reg.fit(X, y)
   >>> print_summary(res)
   <BLANKLINE>
   ==================================================
               MMD Result Summary Report
   ==================================================
   <BLANKLINE>
   Initial Parameters:
       par_v: [0.5 1.5 2.5 3.2]
       par_c: 1.5
   <BLANKLINE>
   Stepsize: 1
   Bandwidth: 1
   <BLANKLINE>
   Estimated Parameters:
        par_v: [1.00998778 1.98323581 3.00644719 3.6917812 ]
   <BLANKLINE>
   Trajectory Summary:
       Number of steps: 1056
       Final trajectory values: [ 1.0100, 1.9832, 3.0064, 3.6918 ]
   <BLANKLINE>
   ==================================================
                     End of Report                  
   ==================================================
   <BLANKLINE>

Compared to the `MMDEstimator` class, now that the model is fitted it is possible to also 
predict with the fitted model.

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> X_test = rng.normal(loc=0, scale=1, size=(10, p))
   >>> mmd_reg.predict(X_test)
   array([6.66346287e+01, 9.92225472e+01, 1.06434759e+02, 6.30142110e-06,
          1.59249268e+01, 1.53260190e-03, 2.60517750e+00, 4.61954372e+05,
          6.08289686e+00, 1.75059655e-01])


Available Models
++++++++++++++++

There are several pre-defined models available in the package, and these should cover most
standard use-cases. The available estimation models can be found at :doc:`../api/regression_models`.

It is also possible to create your own model using the prescribed class 
:py:class:`regmmd.models.base_model.RegressionModel`. A more detailed example can be found Here
:doc:`../example/example3`.
