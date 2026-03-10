Example 1: Estimation
======================

Here is an example of how to use the ``regmmd.estimation`` module. In the
following example the mean and standard deviation of a Gaussian are estimated.

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> from regmmd import MMDEstimator
   >>> from regmmd.utils import print_summary
   >>>
   >>> x = rng.normal(loc=0, scale=1.5, size=(500,))
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
   >>> print_summary(res) 
   <BLANKLINE>
   ==================================================
               MMD Result Summary Report
   ==================================================
   <BLANKLINE>
   Initial Parameters:
       par_v: [-0.07530377  1.29742678]
       par_c: None
   <BLANKLINE>
   Stepsize: 1
   Bandwidth: 1.4351152028386247
   <BLANKLINE>
   Estimated Parameters:
        par_v: [-0.05289928  1.52009151]
   <BLANKLINE>
   Trajectory Summary:
       Number of steps: 1501
       Final trajectory values: [ -0.0529, 1.5201 ]
   <BLANKLINE>
   ==================================================
                     End of Report                  
   ==================================================
   <BLANKLINE>
   
If one wants to estimate only the mean, then the "gaussian-loc" model should be used.

.. doctest::
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   >>> from regmmd import MMDEstimator
   >>> from regmmd.utils import print_summary
   >>>
   >>> x = rng.normal(loc=0, scale=1.5, size=(500,))
   >>> mmd_estim = MMDEstimator(
   ...     model="gaussian-loc",
   ...     par_v=None,
   ...     par_c=1.5,
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
   >>> print_summary(res) 
   <BLANKLINE>
   ==================================================
               MMD Result Summary Report
   ==================================================
   <BLANKLINE>
   Initial Parameters:
       par_v: -0.1388514778320709
       par_c: 1.5
   <BLANKLINE>
   Stepsize: 1
   Bandwidth: 1.3347485118663052
   <BLANKLINE>
   Estimated Parameters:
        par_v: -0.11699387805163043
   <BLANKLINE>
   Trajectory Summary:
       Number of steps: 1501
       Final trajectory values: -0.1170
   <BLANKLINE>
   ==================================================
                     End of Report                  
   ==================================================
   <BLANKLINE>


Available Models
++++++++++++++++

There are several pre-defined models available in the package, and these should cover most
standard use-cases. The available estimation models can be found at :doc:`../api/estimation_models`.

It is also possible to create your own model using the prescribed class 
:py:class:`regmmd.models.base_model.EstimationModel`. A more detailed example can be found Here
:doc:`../example/example3`.