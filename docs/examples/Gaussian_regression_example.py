
import numpy as np

from regmmd import MMDRegressor
from regmmd.models import LinearGaussian


def main():
    n = 1000
    p = 4
    beta = np.arange(1, 5)
    phi = 1

    rng = np.random.default_rng(seed=123)
    print("Sampling X points..")
    X = rng.normal(loc=0, scale=1, size=(n, p))
    noise = rng.normal(0, phi, size=(n,))
    y = 1 + X @ beta + noise

    print("Initializing model")
    par1_init = np.array([0.5, 1.5, 2.5, 3.2])
    par2_init = 2
    model = LinearGaussian(
        beta=par1_init,
        phi=par2_init
    )

    mmd_reg = MMDRegressor(
        model=model,
        par1=par1_init,
        par2=par2_init, 
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
    print(res)

    y_hat = mmd_reg.predict(X)


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)  # Show top 10 time-consuming functions
