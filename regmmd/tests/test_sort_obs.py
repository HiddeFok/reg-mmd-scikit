import numpy as np

from regmmd.optimizer import sort_obs

if __name__ == "__main__":
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 3))

    X_sorted = sort_obs(X)
    print(X_sorted)