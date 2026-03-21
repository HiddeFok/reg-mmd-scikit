.. RegMMD Scikit-Learn Implementation documentation master file, created by
   sphinx-quickstart on Wed Feb 25 17:03:38 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RegMMD Scikit-Learn Implementation documentation
================================================

This is a *scikit-learn* style implementation of the estimation and regression
procedures based one the *Maximum Mean Discrepancy (MMD)* principle between the sample
and the statistical model. These procedures are provable universally consistent and 
extremely robust to outliers. For more information about the theoretical side, the relevant papers
are:

#. `Universal robust regression via maximum mean discrepancy <https://academic.oup.com/biomet/article/111/1/71/7159184>`_, 
    Pierre Alquier, Mathieu Gerber, 2024
#. `Finite sample properties of parametric MMD estimation: Robustness to misspecification and dependence <https://projecteuclid.org/journals/bernoulli/volume-28/issue-1/Finite-sample-properties-of-parametric-MMD-estimation--Robustness-to/10.3150/21-BEJ1338.full>`_,
    Badr-Eddine Chérief-Abdellatif, Pierre Alquier, 2022

Existing R package
++++++++++++++++++

One top of this implementation, there exists an implementation in the R language
by the original authors of the papers, `R package link
<https://cran.r-project.org/web/packages/regMMD/>`_. Most functions in this
package are derived from the R implementation. However, the way the statistical
models are implemented are different in this version to allow users to quickly
implement their own custom model.

Mathematical introduction 
+++++++++++++++++++++++++

In short, the estimator is based on the minimum distance estimator :math:`\hat{\theta}`, which
is defined as 

.. math::
   \hat{\theta} \in \underset{\theta \in \Theta}{\text{argmin}}\;
   d(P_\theta, \hat{P}_n),

for some statistical model :math:`\left\{P_\theta \mid \theta \in \Theta\right\}` and the empirical distrbution
:math:`\hat{P}_n = \frac{1}{n} \sum_{i=1}^n \delta_{X_i}`.

The MMD estimator and regression procedure are derived from taking a specific
distance metric in the above definition.  The distances metric comes from
embedding the distributions in an RKHS :math:`\mathcal{H}` using the feature max
:math:`\varphi \colon : \mathcal{X} \to\mathcal{H}`:

.. math::

   D(P, Q) &:= \|\mu(P) - \mu(Q)\|_\mathcal{H}.\\
   \mu(P) &:= \mathbb{E}_{X \sim P}\left[\varphi(X)\right].

Letting :math:`k(x, y) = \langle \varphi(x), \varphi(y) \rangle_\mathcal{H}`, the actual distance
can be calculated through:

.. math::

   D^2(P, Q) = \mathbb{E}_{X, X' \sim P}\left[k(X, X')\right]
               - 2\mathbb{E}_{X \sim P, X' \sim Q}\left[k(X, X')\right]
               + \mathbb{E}_{X, X' \sim Q}\left[k(X, X')\right].

This expression is used to calculate gradients or stochastic gradients to do 
the actual optimisation, which is done automatically by the package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   example/index
   optimization/index
   robustness/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`