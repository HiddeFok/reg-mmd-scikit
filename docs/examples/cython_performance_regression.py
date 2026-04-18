import numpy as np
import time

from regmmd import MMDRegressor
from regmmd.models.regression.linear_gaussian import LinearGaussianLoc


def benchmark(func, true_beta, n_runs=10):
    """Run func n_runs times and return (mean, std) of elapsed times in ms
    together with (mean, std) of the L1 estimator error against true_beta."""
    times = []
    errors = []
    for _ in range(n_runs):
        start = time.perf_counter()
        est = func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        errors.append(np.abs(est - true_beta).sum())

    times = np.array(times)
    errors = np.array(errors)
    return times.mean(), times.std(), errors.mean(), errors.std()


def print_row(
    n_label,
    mean_np,
    std_np,
    mean_cy,
    std_cy,
    mean_err_np,
    std_err_np,
    mean_err_cy,
    std_err_cy,
):
    speedup = mean_np / mean_cy if mean_cy > 0 else float("inf")
    error_str = "Est Error"
    print(
        f"{n_label:>12}  "
        f"{mean_np:>10.3f} +/- {std_np:<5.2f}"
        f"{mean_cy:>10.3f} +/- {std_cy:<5.2f}"
        f"{speedup:>7.2f}x"
    )
    print(
        f"{error_str:>12} "
        f"{mean_err_np:>11.3f} +/- {std_err_np:<5.2f}"
        f"{mean_err_cy:>10.3f} +/- {std_err_cy:<5.2f}"
    )


def header():
    print(f"{'n':>12}  {'NumPy (ms)':>20}  {'Cython (ms)':>17}  {'Speedup':>6}")
    print("-" * 60)


def bench_sgd_regression():
    print("\n\nSGD Regression: LinearGaussianLoc model")
    print("=" * 60)

    sizes = [100, 500, 10_000, 50_000]
    p = 4
    true_beta = np.arange(1, p + 1, dtype=float)
    phi = 1.0
    beta_init = np.array([0.5, 1.5, 2.5, 3.2])
    phi_init = 2.0

    burn_in = 5000
    n_step = 10000
    stepsize = 1.0
    bandwidth_y = 1.0
    epsilon = 1e-4
    n_runs = 10

    print(f"burn_in={burn_in}, n_step={n_step}  (averaged over {n_runs} runs)")
    header()

    for n in sizes:
        rng = np.random.default_rng(seed=n)
        X = rng.normal(loc=0, scale=1, size=(n, p))
        noise = rng.normal(0, phi, size=(n,))
        y = X @ true_beta + noise

        def make_regressor():
            model = LinearGaussianLoc(
                par_v=beta_init.copy(), par_c=phi_init, random_state=n
            )
            return MMDRegressor(
                model=model,
                fit_intercept=False,
                par_v=beta_init.copy(),
                par_c=phi_init,
                bandwidth_X=0,
                bandwidth_y=bandwidth_y,
                kernel_y="Gaussian",
                solver={
                    "burnin": burn_in,
                    "n_step": n_step,
                    "stepsize": stepsize,
                    "epsilon": epsilon,
                },
            )

        def run_python():
            mmd_reg = make_regressor()
            res = mmd_reg.fit(X=X, y=y, use_exact=False, use_fast=False)
            return res["estimator"]

        def run_cython():
            mmd_reg = make_regressor()
            res = mmd_reg.fit(X=X, y=y, use_exact=False, use_fast=True)
            return res["estimator"]

        mean_np, std_np, mean_err_np, std_err_np = benchmark(
            run_python, true_beta, n_runs
        )
        mean_cy, std_cy, mean_err_cy, std_err_cy = benchmark(
            run_cython, true_beta, n_runs
        )

        print_row(
            f"{n:,}",
            mean_np,
            std_np,
            mean_cy,
            std_cy,
            mean_err_np,
            std_err_np,
            mean_err_cy,
            std_err_cy,
        )


def main():
    bench_sgd_regression()


if __name__ == "__main__":
    main()
