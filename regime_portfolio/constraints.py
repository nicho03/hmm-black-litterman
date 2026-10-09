"""Optimizer constraints: functions f(w, assets) -> list of cvxpy constraints.

Budget (weights sum to 1) and long-only are already enforced by the base class."""
import numpy as np


def max_weight(cap):
    """Maximum weight of any single asset."""
    return lambda w, assets: [w <= cap]


def group_bounds(members, lo=0.0, hi=1.0):
    """Total weight of a group of assets (e.g. an asset class) within [lo, hi]."""
    def f(w, assets):
        mask = np.isin(assets, members).astype(float)
        return [mask @ w >= lo, mask @ w <= hi]
    return f
