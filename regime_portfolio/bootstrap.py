"""Paired circular block bootstrap for the Sharpe ratio.

Each replicate is a "synthetic history" as long as the sample, made of blocks of consecutive
days drawn at random (with replacement). The same blocks are used for every strategy, so
Sharpe differences account for the fact that strategies live through the same days.
It measures the uncertainty due to the market history, not the model's estimation risk."""
import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def sharpe(x, axis=0):
    """Annualized Sharpe ratio of daily excess returns."""
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1) * np.sqrt(252)


def block_indices(T, block, B, rng):
    """Indices (B, T) of B synthetic histories made of circular blocks of `block` days."""
    starts = rng.integers(0, T, size=(B, int(np.ceil(T / block))))
    return ((starts[:, :, None] + np.arange(block)) % T).reshape(B, -1)[:, :T]


def bootstrap_sharpe(excess, blocks=(20, 60, 120), B=2000, seed=0):
    """Sharpe of every column of `excess` over B replicates, for each block length.
    Returns a long DataFrame with columns: block, strategy, rep, sharpe."""
    X = excess.to_numpy()
    rng = np.random.default_rng(seed)
    frames = []
    for L in blocks:
        idx = block_indices(len(X), L, B, rng)
        sr = np.column_stack([sharpe(X[idx, j], axis=1) for j in range(X.shape[1])])
        frames.append(pd.DataFrame(sr, columns=excess.columns).rename_axis("rep").reset_index()
                      .melt(id_vars="rep", var_name="strategy", value_name="sharpe").assign(block=L))
    return pd.concat(frames, ignore_index=True)[["block", "strategy", "rep", "sharpe"]]


def pairwise_tests(boot, observed):
    """Pairwise Sharpe differences, per block length: 95% CI and two-sided p-value of the
    centered bootstrap (H0: zero difference), with Holm correction for multiple comparisons.

    boot     : output of bootstrap_sharpe
    observed : Series {strategy: observed Sharpe}"""
    out = []
    for L, g in boot.groupby("block"):
        wide = g.pivot(index="rep", columns="strategy", values="sharpe")
        rows = []
        for a, b in itertools.combinations(wide.columns, 2):
            d_hat, d = observed[a] - observed[b], wide[a] - wide[b]
            rows.append({"block": L, "A": a, "B": b, "diff_sharpe": d_hat,
                         "ci_low": d.quantile(0.025), "ci_high": d.quantile(0.975),
                         "p": np.mean(np.abs(d - d_hat) >= abs(d_hat))})
        rows = pd.DataFrame(rows)
        rows["p_holm"] = multipletests(rows["p"], method="holm")[1]
        out.append(rows)
    out = pd.concat(out, ignore_index=True)
    out["significant"] = out["p_holm"] < 0.05
    return out
