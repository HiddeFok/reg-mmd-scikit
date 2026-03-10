Estimation Robustness 
=====================

The benefit of this estimator over the standard MLE estimator is that it comes
with robustness guarantees to outliers or contaminated data. As an example, we
estimate the mean for a Gaussian random variable.

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> from regmmd import MMDEstimator
   >>> from regmmd.utils import print_summary
   >>>
   >>> x = rng.normal(loc=0, scale=1.5, size=(50,))
   >>>
   >>> # We contaminate  only one sample
   >>> x[42] = 100
   >>>
   >>> mmd_estim = MMDEstimator(
   ...     model="gaussian",
   ...     par_v=None,
   ...     par_c=None,
   ...     kernel="Gaussian",
   ...     solver={"type": "GD",
   ...             "burnin": 500,
   ...             "n_step": 1000,
   ...             "stepsize": 1, 
   ...             "epsilon": 1e-4
   ...      },
   ...      random_state=42
   ... )
   >>> res = mmd_estim.fit(X=x)  
   >>> print(np.mean(x))
   2.153118443631034
   >>> print(res["estimator"][0])
   0.07003082198862483