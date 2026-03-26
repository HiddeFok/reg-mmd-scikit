import numpy as np
import time

from regmmd.kernels import K1d_dist as K1d_dist_numpy, K1d as K1d_numpy
from regmmd.optimizers._cy_kernels import py_K1d_dist, py_K1d, py_K1d_sym


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
    from regmmd.optimizers._sgd import _sgd_estimation
    from regmmd.optimizers._cy_sgd import cy_sgd_estimation
    from regmmd.models._cy_models import CyGaussianLoc
    from numpy.random import PCG64

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
    n_runs = 10

    rng = np.random.default_rng(seed=42)

    print(f"burn_in={burn_in}, n_step={n_step}  (averaged over {n_runs} runs)")
    header(n_runs)

    for n in sizes:
        X = rng.normal(loc=true_loc, scale=true_scale, size=n)

        def run_python():
            model = GaussianLoc(par_v=0.0, par_c=true_scale, random_state=123)
            _sgd_estimation(
                X=X,
                par_v=np.array([0.0]),
                model=model,
                kernel="Gaussian",
                burn_in=burn_in,
                n_step=n_step,
                stepsize=stepsize,
                bandwidth=bandwidth,
                epsilon=epsilon,
            )

        def run_cython():
            bit_gen = PCG64(seed=123)
            cy_model = CyGaussianLoc(0.0, true_scale, bit_gen)
            cy_sgd_estimation(
                X=X,
                par_v=np.array([0.0]),
                par_c=np.array([true_scale]),
                model=cy_model,
                kernel=KERNEL_MAP["Gaussian"],
                burn_in=burn_in,
                n_step=n_step,
                stepsize=stepsize,
                bandwidth=bandwidth,
                epsilon=epsilon,
            )

        mean_np, std_np = benchmark(run_python, n_runs)
        mean_cy, std_cy = benchmark(run_cython, n_runs)

        print_row(f"{n:,}", mean_np, std_np, mean_cy, std_cy)


def main():
    bench_sgd_estimation()


if __name__ == "__main__":
    main()
