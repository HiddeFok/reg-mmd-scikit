import numpy as np

from regmmd import MMDRegressor
from regmmd.models import GammaRegressionLoc
from regmmd.utils import print_summary


def main():
    n = 10000
    p = 4
    beta = np.arange(1, 5)
    model_true = GammaRegressionLoc(
        par_v=beta, 
        par_c=1,
        random_state=12
    )

    rng = np.random.default_rng(seed=123)
    X = rng.normal(loc=0, scale=1, size=(n, p))
    mu_given_x = model_true.predict(X=X)
    y = model_true.sample_n(n=n, mu_given_x=mu_given_x)

    beta_init = np.array([0.5, 1.5, 2.5, 3.2])
    model = GammaRegressionLoc(
        par_v=beta_init, 
        par_c=1
    )

    mmd_reg = MMDRegressor(
        model=model,
        par_v=beta_init,
        par_c=None,
        fit_intercept=False,
        bandwidth_X=0,
        bandwidth_y=1,
        kernel_y="Gaussian",
        solver={
            "type": "SGD",
            "burnin": 500,
            "n_step": 10000,
            "stepsize": 1,
            "epsilon": 1e-8,
        },
    )

    res = mmd_reg.fit(X, y)
    print_summary(res)

if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
    # Show top 10 time-consuming functions
