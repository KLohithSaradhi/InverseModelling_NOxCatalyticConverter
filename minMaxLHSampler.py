import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc


def maximin_lhs(lower_bounds, upper_bounds, n_samples, n_candidates=1000, rng=None):
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    if lower_bounds.ndim != 1 or upper_bounds.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds must be 1D arrays.")
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("lower_bounds and upper_bounds must have the same length.")
    if np.any(upper_bounds <= lower_bounds):
        raise ValueError("Each upper bound must be strictly greater than lower bound.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1.")

    d = len(lower_bounds)

    # make one numpy generator for reproducibility
    base_rng = np.random.default_rng(rng)

    best_min_dist = -np.inf
    X_best_unit = None

    for _ in range(n_candidates):
        # draw an integer seed for this candidate
        seed_i = int(base_rng.integers(0, 2**32 - 1))

        sampler = qmc.LatinHypercube(d=d, seed=seed_i)
        X_unit = sampler.random(n=n_samples)

        min_dist = pdist(X_unit).min()

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            X_best_unit = X_unit.copy()

    X_best = qmc.scale(X_best_unit, lower_bounds, upper_bounds)
    return X_best