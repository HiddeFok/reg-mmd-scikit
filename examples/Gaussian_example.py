import numpy as np
from regmmd import MMDEstimator
from regmmd.models import GaussianLoc, GaussianScale, Gaussian

rng = np.random.default_rng(seed=123)

x = rng.normal(loc=0, scale=1.5, size=50)

model = GaussianLoc(
    par2 = 1.5
)


mmd_estim = MMDEstimator(
    model=model,
    par1=None, 
    par2=1.5,
    kernel="gaussian",
    solver={
        "type": "GD",
        "burnin": 500,
        "n_step": 1000,
        "stepsize": 1,
        "epsilon": 1e-4
    }
)

res = mmd_estim.fit(
    X=x
)
print(res)