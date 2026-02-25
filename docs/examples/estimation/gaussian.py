import numpy as np

from regmmd import MMDEstimator
from regmmd.models import Gaussian, GaussianLoc
from regmmd.utils import print_summary


def main():
    print("Estimating only the mean")
    rng = np.random.default_rng(seed=123)

    print("Sampling points..")
    x = rng.normal(loc=0, scale=1.5, size=500)

    print("Initializing model")
    model = GaussianLoc(par_c=1.5)

    print("Initializing estimator")
    mmd_estim = MMDEstimator(
        model=model,
        par_v=None,
        par_c=1.5,
        kernel="Gaussian",
        solver={
            "type": "GD",
            "burnin": 5000,
            "n_step": 10000,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )
    print("fitting estimator")
    res = mmd_estim.fit(X=x)
    print_summary(res)

    print("Estimating both the mean and variance")
    #
    print("Sampling points..")
    x = rng.normal(loc=0, scale=1.5, size=50)
    print("Initializing model")
    model = Gaussian(par_v=np.array([0.2, 1.2]), random_state=10)
    print("Initializing estimator")
    mmd_estim = MMDEstimator(
        model=model,
        par_v=np.array([0.2, 1.2]),
        par_c=None,
        kernel="Gaussian",
        solver={
            "type": "SGD",
            "burnin": 500,
            "n_step": 1000,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )
    print("fitting estimator")
    res = mmd_estim.fit(X=x)
    print_summary(res)


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
    # Show top 10 time-consuming functions
