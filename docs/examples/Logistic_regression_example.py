import numpy as np

from regmmd import MMDRegressor
from regmmd.models import Logistic
from regmmd.utils import print_summary


def main():
    n = 1000
    p_param = 4
    beta = np.arange(1, 5)

    rng = np.random.default_rng(seed=123)

    print("Sampling X points..")
    X = rng.normal(loc=0, scale=1, size=(n, p_param))
    p = 1 / (1 + np.exp(-X @ beta))
    y = rng.binomial(1, p, size=(n,))

    print("Initializing model")
    par_v_init = np.array([0.5, 1.5, 2.5, 3.2])
    model = Logistic(par_v=par_v_init)

    mmd_reg = MMDRegressor(
        model=model,
        par_v=par_v_init,
        par_c=None,
        bandwidth_X=0,
        bandwidth_y=1,
        kernel_y="Gaussian",
        solver={
            "type": "SGD",
            "burnin": 500,
            "n_step": 1000,
            "stepsize": 1,
            "epsilon": 1e-4,
        },
    )

    res = mmd_reg.fit(X, y)
    print_summary(res)

    y_hat = mmd_reg.predict(X)


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)  
    # Show top 10 time-consuming functions
