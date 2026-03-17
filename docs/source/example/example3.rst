.. _example3:

Example 3: Writing your own EstimationModel or RegressionModel
==============================================================

The package provides two abstract base classes that can be subclassed to define
custom statistical models:

- :py:class:`regmmd.models.base_model.EstimationModel` — for unconditional parameter 
    estimation (used with :py:class:`regmmd.MMDEstimator`).
- :py:class:`regmmd.models.base_model.RegressionModel` — for conditional
  regression (used with :py:class:`regmmd.MMDRegressor`). Extends
  ``EstimationModel`` with prediction support.

Both classes follow a ``par_v`` / ``par_c`` convention:

- ``par_v`` — *variable* parameters that are optimized by the solver.
- ``par_c`` — *constant* parameters that are held fixed during optimization.

Required Methods
----------------

EstimationModel
+++++++++++++++

Subclasses of ``EstimationModel`` must implement the following methods:

``sample_n(self, n)``
    Draw ``n`` i.i.d. samples from the distribution under the current
    parameters. Used internally by the MMD objective to simulate from the
    model.

``log_prob(self, x)``
    Return the log-likelihood evaluated at each point in ``x``, an array of
    shape ``(n_samples,)`` or ``(n_samples, n_features)``.

``score(self, x)``
    Return the gradient of the log-likelihood with respect to ``par_v`` at
    each point in ``x``. The output shape must be compatible with ``par_v``:
    a scalar gradient per sample when ``par_v`` is a scalar, or an array of
    shape ``(n_samples, len(par_v))`` when ``par_v`` is a vector.

``_init_params(self, X)``
    Initialize the model parameters from the data ``X`` when ``par_v`` or
    ``par_c`` are ``None``. Should call and return ``_get_params()`` at the end.

``_get_params(self)``
    Return the current ``(par_v, par_c)`` tuple so that the optimizer can
    read the parameter state.

``_project_params(self, par_v)``
    Project ``par_v`` onto the feasible set after each optimizer step. Use
    this to enforce constraints (e.g. positivity of a scale parameter). Return
    ``par_v`` unchanged if no constraints are required.

``update(self, par_v)``
    Write a new value of ``par_v`` back into the model's attributes (e.g.
    ``self.loc = par_v``). Called by the optimizer after each gradient step.

RegressionModel
+++++++++++++++

``RegressionModel`` extends ``EstimationModel``. In addition to all methods
listed above (with signatures adapted for the joint input :math:`(X, y)`), the
following extra method is required:

``sample_n(self, n, mu_given_x)``
    Draw ``n`` samples from the conditional distribution
    :math:`p(y \mid X, \text{par\_v}, \text{par\_c})`, given the predicted mean ``mu_given_x``.

``predict(self, X)``
    Return the predicted conditional mean :math:`\mathbb{E}[Y | X]` under the current
    parameters.

``_init_params(self, X, y)``
    Same role as in ``EstimationModel``, but receives both ``X`` and ``y``.


Implementing a Custom EstimationModel
--------------------------------------

The example below implements a simple Poisson estimation model where the rate
parameter ``lam`` is the only variable parameter. ::

    import numpy as np
    from regmmd.models.base_model import EstimationModel


    class PoissonEstimation(EstimationModel):
        def __init__(self, par_v=None, par_c=None, random_state=None):
            # par_v: lambda (rate), par_c: unused
            self.lam = par_v
            self.rng = np.random.default_rng(seed=random_state)

        def sample_n(self, n):
            return self.rng.poisson(lam=self.lam, size=(n,))

        def log_prob(self, x):
            return x * np.log(self.lam) - self.lam - np.log(np.math.factorial(x))

        def score(self, x):
            # d/d(lam) log p(x | lam) = x / lam - 1
            return x / self.lam - 1

        def _init_params(self, X):
            if self.lam is None:
                self.lam = np.mean(X)
            return self._get_params()

        def _get_params(self):
            return self.lam, None

        def _project_params(self, par_v):
            # Rate must be strictly positive
            return max(1e-6, par_v)

        def update(self, par_v):
            self.lam = par_v

This model can then be passed directly to :py:class:`regmmd.MMDEstimator`::

    from regmmd import MMDEstimator

    mmd_estim = MMDEstimator(
        model=PoissonEstimation(),
        par_v=None,
        par_c=None,
        kernel="Gaussian",
        solver={"type": "GD", "burnin": 200, "n_step": 500,
                "stepsize": 0.1, "epsilon": 1e-4},
        random_state=42,
    )
    res = mmd_estim.fit(X=x)


Implementing a Custom RegressionModel
---------------------------------------

The example below implements a Poisson regression model where
:math:`\mathrm{log}(\mathbb{E}[Y \mid X]) = X^T \beta`, with :math:`\beta` as the only variable parameter. ::

    import numpy as np
    from regmmd.models.base_model import RegressionModel


    class PoissonRegression(RegressionModel):
        def __init__(self, par_v=None, par_c=None, random_state=None):
            # par_v: beta coefficients, par_c: unused
            self.beta = par_v
            self.rng = np.random.default_rng(seed=random_state)

        def predict(self, X):
            # Conditional mean: E[Y | X] = exp(X @ beta)
            return np.exp(X @ self.beta)

        def sample_n(self, n, mu_given_x):
            return self.rng.poisson(lam=mu_given_x, size=(n,))

        def log_prob(self, X, y):
            mu = self.predict(X)
            return y * np.log(mu) - mu  # ignoring log(y!) constant

        def score(self, X, y):
            # d/d(beta) log p(y | X, beta) = X^T (y - mu)
            mu = self.predict(X)
            return X * (y - mu)[:, np.newaxis]

        def _init_params(self, X, y):
            if self.beta is None:
                self.beta = np.zeros(X.shape[1])
            return self._get_params()

        def _get_params(self):
            return self.beta, None

        def _project_params(self, par_v):
            return par_v  # no constraints on beta

        def update(self, par_v):
            self.beta = par_v

This model can then be passed directly to :py:class:`regmmd.MMDRegressor`::

    from regmmd import MMDRegressor

    mmd_reg = MMDRegressor(
        model=PoissonRegression(),
        par_v=None,
        par_c=None,
        fit_intercept=False,
        bandwidth_X=0,
        bandwidth_y=1,
        kernel_y="Gaussian",
        solver={"type": "SGD", "burnin": 200, "n_step": 2000,
                "stepsize": 0.1, "epsilon": 1e-6},
        random_state=42,
    )
    res = mmd_reg.fit(X, y)
