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
        f"{n_label:>12}  "
        f"{mean_np:>10.3f} +/- {std_np:<5.2f}"
        f"{mean_cy:>10.3f} +/- {std_cy:<5.2f}"
        f"{speedup:>7.2f}x"
    )


def header(n_runs):
    print(f"{'n':>12}  {'NumPy (ms)':>20}  {'Cython (ms)':>17}  {'Speedup':>6}")
    print("-" * 60)


def bench_K1d_dist():
    print("K1d_dist: elementwise kernel evaluation")
    print("=" * 60)

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    kernels = ["Gaussian", "Laplace", "Cauchy"]
    bandwidth = 1.5
    n_runs = 50

    rng = np.random.default_rng(seed=42)

    for kernel in kernels:
        print(f"\nKernel: {kernel}  (averaged over {n_runs} runs)")
        header(n_runs)

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

            print_row(f"{n:,}", mean_np, std_np, mean_cy, std_cy)


def bench_K1d():
    print("\n\nK1d: pairwise kernel matrix K(x_i - y_j)")
    print("=" * 60)

    sizes = [50, 100, 500, 1_000, 2_000]
    kernels = ["Gaussian", "Laplace", "Cauchy"]
    bandwidth = 1.5
    n_runs = 50

    rng = np.random.default_rng(seed=42)

    for kernel in kernels:
        print(f"\nKernel: {kernel}  (averaged over {n_runs} runs)")
        header(n_runs)

        kernel_id = KERNEL_MAP[kernel]

        for n in sizes:
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            out = np.empty((n, n))

            mean_np, std_np = benchmark(
                lambda: K1d_numpy(x, y, kernel, bandwidth), n_runs
            )
            mean_cy, std_cy = benchmark(
                lambda: py_K1d(x, y, out, kernel_id, bandwidth), n_runs
            )
            mean_cy_sim, std_cy_sim = benchmark(
                lambda: py_K1d_sym(x, y, out, kernel_id, bandwidth), n_runs
            )

            print_row(f"{n:,}x{n:,}", mean_np, std_np, mean_cy, std_cy)
            print_row(f"{n:,}x{n:,}", mean_np, std_np, mean_cy_sim, std_cy_sim)


def main():
    bench_K1d_dist()
    bench_K1d()


if __name__ == "__main__":
    main()
