![code coverage](https://gist.githubusercontent.com/HiddeFok/73b75fdbeb1fba4b08092b0af7b746f3/raw/coverage_badge.svg)
![License](https://img.shields.io/badge/license-GPLv3-green.svg?style=flat)
![python](https://img.shields.io/badge/python-3.12-blue?style=flat&logo=python)

# Scikit implementation of RegMMD

This is a *scikit-learn* style implementation of the estimation and regression
procedures based one the *Maximum Mean Discrepancy (MMD)* principle between the sample
and the statistical model. These procedures are provable universally consistent and 
extremely robust to outliers. For more information about the theoretical side, the relevant papers
are:

1. [Universal robust regression via maximum mean discrepancy](https://academic.oup.com/biomet/article/111/1/71/7159184), 
    Pierre Alquier, Mathieu Gerber, 2024
2. [Finite sample properties of parametric MMD estimation: Robustness to misspecification and dependence](https://projecteuclid.org/journals/bernoulli/volume-28/issue-1/Finite-sample-properties-of-parametric-MMD-estimation--Robustness-to/10.3150/21-BEJ1338.full),
    Badr-Eddine Chérief-Abdellatif, Pierre Alquier, 2022

## Existing R package

One top of this implementation, there exists an implementation in the R language
by the original authors of the papers, 
[R package link](https://cran.r-project.org/web/packages/regMMD/). 
Most functions in this package are derived from the R implementation. However,
the way the statistical models are implemented are different in this version to
allow users to quickly implement their own custom model.


## Installation

This package is developed for Python 3.12+ and can be installed by cloning this repository and installing through `pip` or `uv`. The package will be released
as an official PyPI package, when it is finished.

### Using pip
```bash
pip install regmmd
```

### Using uv
Assuming that you are in a directory where a `uv` project is initialised. 
```bash
uv add regmmd
```

## Optimization

The optimizer uses exact analytical gradient methods wherever possible, and
falls back to a general stochastic gradient descent (SGD) method otherwise.
Each model can define an `_exact_fit()` method that computes closed-form
gradients for its specific combination of model and kernel. When an exact
method is not available (or the model/kernel combination is not supported),
the optimizer automatically falls back to the general-purpose SGD solver which
works with any model. For a detailed overview, see the
[documentation](https://hiddefok.github.io/reg-mmd-scikit/).

## Examples

The package has 2 main classes, `MMDEstimator` and `MMDRegressor` that
implement the estimation and regression procedures respectively. They follow
a scikit-learn style implementation. A list of all available models and a description of how to implement your own model, can be found on the [documentation
site](https://hiddefok.github.io/reg-mmd-scikit/).

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
