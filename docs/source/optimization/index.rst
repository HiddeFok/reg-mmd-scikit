Optimization
============

The optimization procedure in RegMMD follows a two-tier strategy: it uses
**exact analytical gradient methods** when available, and falls back to a
**general-purpose stochastic gradient descent (SGD)** method otherwise.

Exact methods vs. SGD
+++++++++++++++++++++

Exact methods
-------------

For certain combinations of statistical model and kernel, the MMD objective and
its gradient can be computed in closed form, without resorting to Monte Carlo
sampling.  This has two key advantages:

- **No sampling variance**: the gradient is deterministic, leading to more
  stable optimization.
- **Efficiency**: direct computation avoids the cost of drawing and evaluating
  random samples at each step.

Each model class can optionally implement an ``_exact_fit()`` method.  When
called, the optimizer first tries this method.  If it returns a result, that
result is used directly.  If it returns ``None`` (meaning the current
model/kernel combination has no exact implementation), the optimizer falls back
to SGD.

The decision logic in the estimator and regressor looks like this:

.. code-block:: python

   # 1. Try the exact method
   res = model._exact_fit(X=X, ...)

   # 2. Fall back to SGD if no exact method is available
   if res is None:
       res = _sgd_estimation(X=X, ...)

SGD fallback
------------

The general SGD solver works with **any** model and kernel combination.  It
approximates the MMD gradient by sampling from the model at each iteration and
uses AdaGrad with Polyak-Ruppert averaging for stable convergence.

For the regression setting, two SGD variants are available:

- **Tilde estimator** (``_sgd_tilde_regression``): uses only a kernel on
  :math:`Y`.  This is selected when no covariate kernel is specified
  (``bandwidth_X = 0``).
- **Hat estimator** (``_sgd_hat_regression``): uses a product kernel on
  :math:`(X, Y)`.  This is selected when a covariate kernel is specified
  (``bandwidth_X > 0``).

Available exact methods
+++++++++++++++++++++++

The table below summarises which model/kernel combinations currently have exact
methods implemented.

Estimation
----------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Kernel
     - Method
   * - ``GaussianLoc``
     - Gaussian
     - Exact gradient descent with AdaGrad + Polyak-Ruppert averaging

All other estimation models (``GaussianScale``, ``Gaussian``, ``Beta``,
``Poisson``, ``Gamma``, etc.) use the general SGD solver.

Regression
----------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Model
     - Kernel
     - Estimator
     - Method
   * - ``LinearGaussianLoc``
     - Gaussian
     - Tilde
     - Exact GD with backtracking line search
   * - ``LinearGaussian``
     - Gaussian
     - Tilde
     - Exact GD with backtracking line search (joint optimisation of coefficients and variance)
   * - ``Logistic``
     - Any
     - Tilde
     - Exact GD with backtracking line search
   * - ``Logistic``
     - Any
     - Hat
     - Exact AdaGrad (analytical gradients over :math:`Y \in \{0, 1\}`)

All other regression models (``GammaRegressionLoc``,
``PoissonRegressionLoc``, etc.) use the general SGD solver.

Implementing a custom exact method
+++++++++++++++++++++++++++++++++++

To add an exact method for a new model, override the ``_exact_fit()`` method in
your model class.  The base class implementation returns ``None``, which
triggers the SGD fallback.  Your override should:

1. Check whether the kernel and other settings are supported by your exact
   implementation.
2. If supported, run the optimization and return the result dictionary.
3. If not supported, return ``None`` to fall back to SGD.

.. code-block:: python

   class MyModel(BaseModel):
       def _exact_fit(self, X, par_v, par_c, solver, kernel, bandwidth):
           if kernel != "Gaussian":
               return None  # fall back to SGD

           # ... compute exact gradients and optimize ...
           return {"estimator": par_v_opt, "trajectory": trajectory}
