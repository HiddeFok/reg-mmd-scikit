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
    Badr-Eddine Chérief-Abdellatif, Pierre Alquier, 2022
2. [Finite sample properties of parametric MMD estimation: Robustness to misspecification and dependence](https://projecteuclid.org/journals/bernoulli/volume-28/issue-1/Finite-sample-properties-of-parametric-MMD-estimation--Robustness-to/10.3150/21-BEJ1338.full),
    Badr-Eddine Chérief-Abdellatif, Pierre Alquier, 2022


## Installation

This package is developed for Python 3.12+ and can be installed by cloning this repository and installing through `pip`.

```bash
git clone git@github.com:HiddeFok/reg-mmd-scikit.git
pip install -e .
```

## Examples

The package has 2 main classes, `MMDEstimator` and `MMDRegressor` that
implement the estimation and regression procedures respectively. They follow
a scikit-learn style implementation

### Estimation
```python
from regmmd import MMDestimator
import numpy as np

rng = np.random.default_rng(seed=123)
x = rng.normal(loc=0, scale=1.5, size=500)

mmd_estim = MMDEstimator(
    model="gaussian-loc",
    par_v=None,
    par_c=1.5,
    kernel="Gaussian",
    solver={
        "type": "GD",
        "burnin": 5000,
        "n_step": 10000,
        "stepsize": 1,
        "epsilon": 1e-4,
    }
)
res = mmd_estim.fit(X=x)
```

### Regression
```python
from regmmd import MMDRegressor
import numpy as np

rng = np.random.default_rng(seed=123)

n = 10000
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
    par_c=None,
    fit_intercept=False,
    bandwidth_X=1,
    bandwidth_y=1,
    kernel_y="Laplace",
    solver={
        "type": "SGD",
        "burnin": 500,
        "n_step": 10000,
        "stepsize": 1,
        "epsilon": 1e-8,
    },
)

res = mmd_reg.fit(X, y)
```

