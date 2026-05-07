import numpy as np
import time

from regmmd import MMDEstimator

KERNEL_MAP = {"Gaussian": 0, "Laplace": 1, "Cauchy": 2}


def benchmark(func, n_runs=50):
    """Run func n_runs times and return (mean, std) of elapsed times in ms."""
    times = []
    estimator = []
    for _ in range(n_runs):
        start = time.perf_counter()
        est = func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        estimator.append(abs(est - np.array([2.0, 1.5])).sum())

    times = np.array(times)
    estimator = np.array(estimator)
    return times.mean(), times.std(), estimator.mean(), estimator.std()


def print_row(
    n_label,
    mean_np,
    std_np,
    mean_cy,
    std_cy,
    mean_est_np,
    std_est_np,
    mean_est_cy,
    std_est_cy,
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
        f"{mean_est_np:>11.3f} +/- {std_est_np:<5.2f}"
        f"{mean_est_cy:>10.3f} +/- {std_est_cy:<5.2f}"
    )


def header(n_runs):
    print(f"{'n':>12}  {'NumPy (ms)':>20}  {'Cython (ms)':>17}  {'Speedup':>6}")
    print("-" * 60)


def bench_sgd_estimation():
    from regmmd.models.estimation.gaussian import Gaussian

    print("\n\nSGD Estimation: Gaussian model")
    print("=" * 60)

    sizes = [100, 500, 1_000, 2000]
    burn_in = 500
    n_step = 1000
    stepsize = 1.0
    bandwidth = 2.0
    epsilon = 1e-4
    true_loc = 2.0
    true_scale = 1.5
    n_runs = 10

    print(f"burn_in={burn_in}, n_step={n_step}  (averaged over {n_runs} runs)")
    header(n_runs)

    for n in sizes:
        rng = np.random.default_rng(seed=n)
        X = rng.normal(loc=true_loc, scale=true_scale, size=n)

        def run_python():
            model = Gaussian(par_v=np.array([0.0, 2.0]), par_c=None, random_state=n)
            mmd_estim = MMDEstimator(
                model=model,
                par_v=np.array([0.0, 2.0]),
                par_c=None,
                kernel="Gaussian",
                bandwidth=bandwidth,
                solver={
                    "burnin": burn_in,
                    "n_step": n_step,
                    "stepsize": stepsize,
                    "epsilon": epsilon,
                },
            )
            res = mmd_estim.fit(X=X, use_exact=False, use_fast=False)
            return res["estimator"]

        def run_cython():
            model = Gaussian(par_v=np.array([0.0, 2.0]), par_c=None, random_state=n)
            mmd_estim = MMDEstimator(
                model=model,
                par_v=np.array([0.0, 2.0]),
                par_c=None,
                kernel="Gaussian",
                bandwidth=bandwidth,
                solver={
                    "burnin": burn_in,
                    "n_step": n_step,
                    "stepsize": stepsize,
                    "epsilon": epsilon,
                },
            )
            res = mmd_estim.fit(X=X, use_exact=False, use_fast=True)
            return res["estimator"]

        mean_np, std_np, mean_est_np, std_est_np = benchmark(run_python, n_runs)
        mean_cy, std_cy, mean_est_cy, std_est_cy = benchmark(run_cython, n_runs)

        print_row(
            f"{n:,}",
            mean_np,
            std_np,
            mean_cy,
            std_cy,
            mean_est_np,
            std_est_np,
            mean_est_cy,
            std_est_cy,
        )


def main():
    bench_sgd_estimation()


if __name__ == "__main__":
    main()
