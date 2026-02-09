import numpy as np
from regmmd import MMDEstimator
from regmmd.models import GaussianLoc, GaussianScale, Gaussian

import cProfile
import pstats


def main():
    print("Estimating only the mean")
    rng = np.random.default_rng(seed=123)
    print("Sampling points..")
    x = rng.normal(loc=0, scale=1.5, size=50)
    print("Initializing model")
    model = GaussianLoc(
        par2 = 1.5
    )
    print("Initializing estimator")
    mmd_estim = MMDEstimator(
        model=model,
        par1=None, 
        par2=1.5,
        kernel="Gaussian",
        solver={
            "type": "GD",
            "burnin": 500,
            "n_step": 1000,
            "stepsize": 1,
            "epsilon": 1e-4
        }
    )
    print("fitting estimator")
    res = mmd_estim.fit(
        X=x
    )
    print(res)

    print("Estimating both the mean and variance")
    # 
    print("Sampling points..")
    x = rng.normal(loc=0, scale=1.5, size=50)
    print("Initializing model")
    model = Gaussian(par1=0.2, par2=1.2, random_state=10)
    print("Initializing estimator")
    mmd_estim = MMDEstimator(
        model=model,
        par1=1.0, 
        par2=0.2,
        kernel="Gaussian",
        solver={
            "type": "SGD",
            "burnin": 500,
            "n_step": 1000,
            "stepsize": 1,
            "epsilon": 1e-4
        }
    )
    print("fitting estimator")
    res = mmd_estim.fit(
        X=x
    )
    print(res)

if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)  # Show top 10 time-consuming functions