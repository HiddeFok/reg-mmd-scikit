import numpy as np
import time

from regmmd.kernels import K1d_dist as K1d_dist_numpy
from regmmd.optimizers._cy_kernels import py_K1d_dist


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


def main():
    print("Cython vs NumPy performance comparison for K1d_dist")
    print("=" * 55)

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    kernels = ["Gaussian", "Laplace", "Cauchy"]
    bandwidth = 1.5
    n_runs = 50

    rng = np.random.default_rng(seed=42)

    for kernel in kernels:
        print(f"\nKernel: {kernel}  (averaged over {n_runs} runs)")
        print(f"{'n':>10}  {'NumPy (ms)':>14}  {'Cython (ms)':>14}  {'Speedup':>8}")
        print("-" * 55)

        kernel_id = KERNEL_MAP[kernel]

        for n in sizes:
            u = rng.standard_normal(n)
            out = np.empty(n)

            mean_np, std_np = benchmark(
                lambda: K1d_dist_numpy(u, kernel, bandwidth), n_runs
            )
            mean_cy, std_cy = benchmark(
                lambda: py_K1d_dist(u, out, kernel_id, bandwidth), n_runs
            )

            speedup = mean_np / mean_cy if mean_cy > 0 else float("inf")
            print(
                f"{n:>10,}  "
                f"{mean_np:>10.3f} +/- {std_np:<5.2f}"
                f"{mean_cy:>10.3f} +/- {std_cy:<5.2f}"
                f"{speedup:>7.2f}x"
            )


if __name__ == "__main__":
    main()
