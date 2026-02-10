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

The package has 2 main classes, `MMDEstimator` and `MMDRegression` that
implement the estimation and regression procedures respectively. They follow
a scikit-learn style implementation

### Estimation
```python
from regmmd import MMDestimator
```

### Regression
```python
from regmmd import MMDRegression
```

