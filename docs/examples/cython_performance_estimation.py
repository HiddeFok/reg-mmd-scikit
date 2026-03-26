import numpy as np
import time

from regmmd import MMDEstimator

KERNEL_MAP = {"Gaussian": 0, "Laplace": 1, "Cauchy": 2}


def benchmark(func, n_runs=50):
    """Run func n_runs times and return (mean, std) of elapsed times in ms."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times = np.array(times)
    return times.mean(), times.std()


def print_row(n_label, mean_np, std_np, mean_cy, std_cy):
    speedup = mean_np / mean_cy if mean_cy > 0 else float("inf")
    print(
        f"{n_label:>10}  "
        f"{mean_np:>10.3f} +/- {std_np:<5.2f}"
        f"{mean_cy:>10.3f} +/- {std_cy:<5.2f}"
        f"{speedup:>7.2f}x"
    )


def header(n_runs):
    print(f"{'n':>10}  {'NumPy (ms)':>14}  {'Cython (ms)':>14}  {'Speedup':>8}")
    print("-" * 55)

def bench_sgd_estimation():
    from regmmd.models.estimation.gaussian import GaussianLoc

    print("\n\nSGD Estimation: GaussianLoc model")
    print("=" * 55)

    sizes = [100, 500, 1_000]
    burn_in = 500
    n_step = 1000
    stepsize = 1.0
    bandwidth = 2.0
    epsilon = 1e-4
    true_loc = 1.0
    true_scale = 1.5
    n_runs = 2

    rng = np.random.default_rng(seed=42)

    print(f"burn_in={burn_in}, n_step={n_step}  (averaged over {n_runs} runs)")
    header(n_runs)

    for n in sizes:
        X = rng.normal(loc=true_loc, scale=true_scale, size=n)

        def run_python():
            model = GaussianLoc(par_v=0.0, par_c=true_scale, random_state=123)
            mmd_estim = MMDEstimator(
                model=model,
                par_v=np.array([0.0]),
                par_c=np.array([true_scale]),
                kernel="Gaussian",
                bandwidth=bandwidth,
                solver={
                    "burnin": burn_in,
                    "n_step": n_step,
                    "stepsize": stepsize,
                    "epsilon": epsilon
                }
            )
            res = mmd_estim.fit(
                X=X, 
                use_exact=False,
                use_fast=False
            )

        def run_cython():
            model = GaussianLoc(par_v=0.0, par_c=true_scale, random_state=123)
            mmd_estim = MMDEstimator(
                model=model,
                par_v=np.array([0.0]),
                par_c=np.array([true_scale]),
                kernel="Gaussian",
                bandwidth=bandwidth,
                solver={
                    "burnin": burn_in,
                    "n_step": n_step,
                    "stepsize": stepsize,
                    "epsilon": epsilon
                }
            )
            res = mmd_estim.fit(
                X=X, 
                use_exact=False,
                use_fast=True
            )

        mean_np, std_np = benchmark(run_python, n_runs)
        mean_cy, std_cy = benchmark(run_cython, n_runs)

        print_row(f"{n:,}", mean_np, std_np, mean_cy, std_cy)


def main():
    bench_sgd_estimation()


if __name__ == "__main__":
    main()
