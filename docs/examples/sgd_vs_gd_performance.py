import numpy as np
import time
from datetime import timedelta

from regmmd import MMDEstimator, MMDRegressor
from regmmd.models import GaussianLoc
from regmmd.models import LinearGaussianLoc
from regmmd.utils import print_summary


def main():
    print("\nSGD vs GD comparison in the GaussianLoc EstimationModel")
    BURNIN = 500
    N_STEPS = 1000

    par_v_init = 1.
    par_c_init = 2.

    rng = np.random.default_rng(seed=123)

    x = rng.normal(loc=0, scale=1.5, size=1000)
    model = GaussianLoc(par_v=par_v_init, par_c=par_c_init, random_state=20)

    mmd_estim = MMDEstimator(
        model=model,
        par_v=par_v_init, 
        par_c=par_c_init,
        kernel="Gaussian",
        bandwidth=2.,
        solver={
            "burnin": BURNIN,
            "n_step": N_STEPS,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )
    start = time.time()
    res = mmd_estim.fit(X=x)
    gd_diff = time.time() - start
    gd_param = res["estimator"]
    print(f"\tTime elapsed using GD: {timedelta(seconds=gd_diff)}")
    print_summary(res)

    model = GaussianLoc(par_v=par_v_init, par_c=par_c_init, random_state=20)

    mmd_estim = MMDEstimator(
        model=model,
        par_v=par_v_init, 
        par_c=par_c_init,
        bandwidth=2.,
        kernel="Gaussian",
        solver={
            "burnin": BURNIN,
            "n_step": N_STEPS,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )
    start = time.time()
    res = mmd_estim.fit(X=x, use_exact=False)
    sgd_diff = time.time() - start
    sgd_param = res["estimator"][0]
    print(f"\tTime elapsed using SGD: {timedelta(seconds=sgd_diff)}")
    print_summary(res)

    print(f"\tGD is {sgd_diff / gd_diff:.6f} times faster")
    print(f"\tGD is {(abs(sgd_param) / abs(gd_param)):.4f} times closer")


    print("\nSGD vs GD comparison in the LinearGaussianLoc RegressionModel")
    n = 5000
    p_param = 4
    beta = np.arange(1, 5)

    rng = np.random.default_rng(seed=123)

    X = rng.normal(loc=0, scale=1, size=(n, p_param))
    X_test = rng.normal(loc=0, scale=1, size=(n, p_param))

    model_true = LinearGaussianLoc(par_v=beta, par_c=1, random_state=24)
    mu = model_true.predict(X=X)
    mu_test = model_true.predict(X=X_test)

    y = model_true.sample_n(n=n, mu_given_x=mu)
    y_test = model_true.sample_n(n=n, mu_given_x=mu_test)

    par_v_init = np.array([0.5, 1.5, 1.1, 1.2])

    model = LinearGaussianLoc(par_v=par_v_init, par_c=1, random_state=10)

    mmd_reg = MMDRegressor(
        model=model,
        par_v=par_v_init,
        par_c=None,
        bandwidth_X=0,
        bandwidth_y=2,
        kernel_y="Gaussian",
        fit_intercept=False,
        solver={
            "n_step": N_STEPS + BURNIN,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )
    start = time.time()
    res = mmd_reg.fit(X, y)
    gd_diff = time.time() - start

    y_pred = mmd_reg.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"\tGD convergence:{res["convergence"]}")
    print(f"\tGD steps take:{res["trajectory"].shape[1]}")
    print(f"\tGD MSE: {mse:.4f}")
    print(f"\tGD param:")
    print(res["estimator"])

    model = LinearGaussianLoc(par_v=par_v_init, par_c=1, random_state=10)

    mmd_reg = MMDRegressor(
        model=model,
        par_v=par_v_init,
        par_c=None,
        bandwidth_X=0,
        bandwidth_y=2,
        kernel_y="Cauchy",
        fit_intercept=False,
        solver={
            "burnin": BURNIN,
            "n_step": N_STEPS,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
        random_state=123
    )
    start = time.time()
    res = mmd_reg.fit(X, y, use_exact=False)
    sgd_diff = time.time() - start
    y_pred = mmd_reg.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)

    print(f"\tSGD convergence:{res["convergence"]}")
    print(f"\tSGD steps take:{res["trajectory"].shape[1]}")
    print(f"\tSGD MSE: {mse:.4f}")
    print(f"\tSGD param:")
    print(res["estimator"])


    print(f"\tTime elapsed using GD: {timedelta(seconds=gd_diff)}")
    print(f"\tTime elapsed using SGD: {timedelta(seconds=sgd_diff)}")

    print(f"\tGD is {sgd_diff / gd_diff:.6f} times faster")

if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
    # Show top 10 time-consuming functions
