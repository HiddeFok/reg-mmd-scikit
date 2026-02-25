import numpy as np

from regmmd import MMDRegressor
from regmmd.models import Gamma
from regmmd.utils import print_summary


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
    beta_init = [0.5, 1.5, 2.5, 3.2]
    phi_init = [2.0]
    par_v_init = np.array(beta_init + phi_init)
    par_c_init = None
    model = LinearGaussian(par_v=par_v_init, par_c=par_c_init)

    mmd_reg = MMDRegressor(
        model=model,
        par_v=par_v_init,
        par_c=par_c_init,
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

    print("Some test statistics")
    X_test = rng.normal(loc=0, scale=1, size=(n, p))
    noise_test = rng.normal(0, phi, size=(n,))
    y_test = X_test @ beta + noise_test
    y_pred = mmd_reg.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mean_constant = np.mean((y_test - y_test.mean()) ** 2)
    print(f"MSE = {mse:.4f}")
    print(f"R2 = {1 - mse / mean_constant:.4f}\n")

    # y_hat = mmd_reg.predict(X)
    print("Doing the hat estimation")
    print("Sampling X points..")
    X = rng.normal(loc=0, scale=1, size=(n, p))
    noise = rng.normal(0, phi, size=(n,))
    y = 1 + X @ beta + noise

    print("Initializing model")
    beta_init = [0.5, 1.5, 2.5, 3.2]
    phi_init = [2.0]
    par_v_init = np.array(beta_init + phi_init)
    par_c_init = None
    model = LinearGaussian(par_v=par_v_init, par_c=par_c_init)

    mmd_reg = MMDRegressor(
        model=model,
        par_v=par_v_init,
        par_c=par_c_init,
        bandwidth_X=1,
        bandwidth_y=1,
        kernel_y="Gaussian",
        kernel_X="Laplace",
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

    print("Some test statistics")
    X_test = rng.normal(loc=0, scale=1, size=(n, p))
    noise_test = rng.normal(0, phi, size=(n,))
    y_test = X_test @ beta + noise_test
    y_pred = mmd_reg.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    mean_constant = np.mean((y_test - y_test.mean()) ** 2)
    print(f"MSE = {mse:.4f}")
    print(f"R2 = {1 - mse / mean_constant:.4f}\n")


if __name__ == "__main__":
    main()
    # cProfile.run("main()", "profile_stats")
    # stats = pstats.Stats("profile_stats")
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
    # Show top 10 time-consuming functions
