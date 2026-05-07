![code coverage](https://gist.githubusercontent.com/HiddeFok/73b75fdbeb1fba4b08092b0af7b746f3/raw/coverage_badge.svg)
![License](https://img.shields.io/badge/license-GPLv3-green.svg?style=flat)
![python](https://img.shields.io/badge/python-3.12-blue?style=flat&logo=python)
![PyPI](https://img.shields.io/pypi/v/regmmd)

# regmmd — robust estimation and regression via MMD

`regmmd` is a scikit-learn-compatible implementation of parametric estimation
and regression using the **Maximum Mean Discrepancy (MMD)** criterion. MMD
measures the distance between two distributions through their mean embeddings
in a reproducing kernel Hilbert space; minimising it between an empirical
sample and a parametric model yields estimators that are **provably universally
consistent and robust to outliers and model misspecification** — including
adversarial contamination of an arbitrary fraction of the data.

The methodology follows:

1. Alquier & Gerber, *Universal robust regression via maximum mean
   discrepancy*, Biometrika (2024).
   [link](https://academic.oup.com/biomet/article/111/1/71/7159184)
2. Chérief-Abdellatif & Alquier, *Finite-sample properties of parametric MMD
   estimation: robustness to misspecification and dependence*, Bernoulli
   (2022).
   [link](https://projecteuclid.org/journals/bernoulli/volume-28/issue-1/Finite-sample-properties-of-parametric-MMD-estimation--Robustness-to/10.3150/21-BEJ1338.full)

## Existing R package

One top of this implementation, there exists an implementation in the R language
by the original authors of the papers, 
[R package link](https://cran.r-project.org/web/packages/regMMD/). 
Most functions in this package are derived from the R implementation. However,
the way the statistical models are implemented are different in this version to
allow users to quickly implement their own custom model.

## Quickstart

```python
import numpy as np
from regmmd import MMDEstimator

rng = np.random.default_rng(0)
X = rng.normal(loc=2.0, scale=1.5, size=500)

est = MMDEstimator(
    model="gaussian-loc",
    par_c=1.5, 
    kernel="Gaussian",
    solver={
        "burnin": 500,
        "n_step": 1000, 
        "stepsize": 1.0,
        "epsilon": 1e-4
        }
    )
res = est.fit(X)
print(res["estimator"])
```

## Installation

This package is developed for Python 3.11+ and can be
installed through `pip` or `uv`. 

### Using pip
```bash
pip install regmmd
```

### Using uv
```bash
uv add regmmd
```

To work from source (e.g. for development), see
[Development](#development).

## What's in the package

Two scikit-learn-style estimators:

- **`MMDEstimator`** — fits a univariate parametric model to an i.i.d. sample
  by minimising MMD between the empirical and model distributions.
- **`MMDRegressor`** — fits a regression model (`fit`/`predict` API,
  `BaseEstimator` subclass — composes with `Pipeline`, `GridSearchCV`, etc.) by
  minimising MMD between observed and model-predicted conditional
  distributions.

  ### Supported models

**Estimation** (`regmmd.models`): 
`Gaussian`, `GaussianLoc`, `GaussianScale`, `Cauchy`, `Beta`, `BetaA`, `BetaB`,
`Gamma`, `GammaShape`, `GammaRate`, `Binomial`, `Poisson`, `Geometric`,
`Pareto`, `Dirac`, `ContinuousUniformLoc`, `ContinuousUniformUpper`,
`ContinuousUniformLowerUpper`, `DiscreteUniform`.

**Regression** (`regmmd.models`): 
`LinearGaussian`, `LinearGaussianLoc`, `Logistic`, `GammaRegression`,
`GammaRegressionLoc`, `PoissonRegression`, `BetaRegression`,
`BetaRegressionLoc`.

Each model can be selected by string (e.g. `model="gamma-regression-loc"`) or
by passing a class instance. See the
[documentation](https://hiddefok.github.io/reg-mmd-scikit/) for definitions and
parameter conventions, plus a guide to implementing your own model.

### Kernels and bandwidth

Three 1-D kernels are supported for both ``X`` and ``y``: ``"Gaussian"``,
``"Laplace"``, ``"Cauchy"``. Bandwidths can be fixed floats or `"auto"` (the
median heuristic). Setting ``bandwidth_X=0`` in `MMDRegressor` selects the
*tilde* estimator (kernel only on ``y``); a positive bandwidth selects the
*hat* estimator (product kernel on ``(X, y)``). The hat estimator is more
robust to covariate contamination at the cost of preprocessing pairwise
covariate distances.

## Examples

### Estimation
```python
import numpy as np

from regmmd import MMDEstimator
from regmmd.utils import print_summary


rng = np.random.default_rng(seed=123)
X = rng.normal(loc=0, scale=1.5, size=500)

mmd_estim = MMDEstimator(
    model="gaussian-loc",
    par_v=None,
    par_c=1.5,
    kernel="Gaussian",
    solver={
        "burnin": 500,
        "n_step": 1000,
        "stepsize": 1,
        "epsilon": 1e-4,
    }
)
res = mmd_estim.fit(X=X)
print_summary(res)
```

### Regression
```python
from regmmd import MMDRegressor
from regmmd.models import GammaRegressionLoc
from regmmd.utils import print_summary

import numpy as np

rng = np.random.default_rng(seed=123)

n = 1000
p = 4
beta = np.arange(1, 5)
model_true = GammaRegressionLoc(par_v=beta, par_c=1, random_state=12)

X = rng.normal(loc=0, scale=1, size=(n, p))
mu_given_x = model_true.predict(X=X)
y = model_true.sample_n(n=n, mu_given_x=mu_given_x)

beta_init = np.array([0.5, 1.5, 2.5, 3.2])

mmd_reg = MMDRegressor(
    model="gamma-regression-loc",
    par_v=beta_init,
    par_c=1.5,
    fit_intercept=False,
    bandwidth_X=0,
    bandwidth_y=5,
    kernel_y="Gaussian",
    solver={
        "burnin": 5000,
        "n_step": 10000,
        "stepsize": 1,
        "epsilon": 1e-8,
    },
)

res = mmd_reg.fit(X=X, y=y)
print_summary(res)

```

## Optimization

The optimizer uses exact analytical gradient methods wherever possible, and
falls back to a general stochastic gradient descent (SGD) method otherwise.
Each model can define an `_exact_fit()` method that computes closed-form
gradients for its specific combination of model and kernel. When an exact
method is not available (or the model/kernel combination is not supported),
the optimizer automatically falls back to the general-purpose SGD solver which
works with any model. There are also two flags that can enforce one of 
the two speed-ups of the optimization loop:

- `use_exact=True` (default) — try the analytical path first, fall back to SGD.
- `use_fast=True` (default, estimation only) — use the Cython SGD loop (5–10×
faster) when a Cython mirror of the model is available.

For a detailed overview, see the
[documentation](https://hiddefok.github.io/reg-mmd-scikit/).

### Estimation optimization speedup

For the estimation procedure, the SGD optimization loop is also written in 
Cython. To benefit from this speed, you would need to add your model to
the `_cy_estimation_models.pyx` and `_cy_estimation_models.pyd` files
and update the `_build_cy_model` method in the python class. No extra
Cython functionality was added to the regression models, as the optimization
is already quite fast, because of the algorithmic improvements that were
made in the published papers. Specifically, all the required kernel operations
are of order $O(n)$ instead of $O(n^2)$

## Development

Clone and install with the test dependencies via
[`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/HiddeFok/reg-mmd-scikit
cd reg-mmd-scikit
uv sync --group test
```

Run the test suite with coverage:

```bash
uv run pytest tests/ --cov=regmmd
```

Rebuild the Cython extensions (`regmmd/optimizers/_cy_*.pyx`,
`regmmd/models/_cy_estimation_models.pyx`) after editing them:

```bash
uv run python setup.py build_ext --inplace
```

To add a Cython-accelerated mirror of a new estimation model, edit
`_cy_estimation_models.pyx` / `_cy_estimation_models.pxd`, then expose it via
`_build_cy_model` on the Python class.
