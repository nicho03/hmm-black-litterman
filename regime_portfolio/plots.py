"""Result plots."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from .base import CASH
from .strategies import FixedWeights

# Categorical palette in fixed order (colorblind-safe on adjacent pairs)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REGIME_COLORS, REGIME_LABELS = ["#d03b3b", "#b4b2ab", "#0ca30c"], ["Crisis", "Neutral", "Calm"]
CASH_COLOR, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax, title=None):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11)


def _save(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _regime_band(ax, regimes, alpha, label=True):
    """Shade the x-axis by predicted regime (0 = crisis, 1 = neutral, 2 = calm)."""
    for k, (color, name) in enumerate(zip(REGIME_COLORS, REGIME_LABELS)):
        ax.fill_between(regimes.index, 0, 1, where=regimes == k, color=color, alpha=alpha, lw=0,
                        transform=ax.get_xaxis_transform(), step="post",
                        label=f"Regime: {name}" if label else None)


def performance(strategies, regimes, path=None):
    """1) Portfolio value (log scale, net of costs). Background = predicted regime."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _regime_band(ax, regimes, alpha=0.13)
    for i, s in enumerate(strategies):
        passive = isinstance(s, FixedWeights)
        ax.plot(s.wealth.index, s.wealth, color=COLORS[i], lw=1.3 if passive else 1.8,
                ls="--" if passive else "-", label=s.name)
    ax.set_yscale("log")
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.set_xlim(regimes.index[0], regimes.index[-1])
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    _style(ax, "Portfolio value (1 = start of test, net of costs)")
    return _save(fig, path)


def weights_by_group(strategies, groups, path=None):
    """3) Weights over time by asset class, one panel per strategy. Above each panel, a strip
    with the regime predicted by the HMM at every rebalance."""
    n = len(strategies)
    ratios = []
    for i in range(n):
        ratios += [0.09, 1] + ([0.28] if i < n - 1 else [])       # strip, weights, spacer
    fig = plt.figure(figsize=(12, 3.1 * n))
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.04)
    colors = COLORS[:len(groups)] + [CASH_COLOR]
    first_ax = None
    for i, s in enumerate(strategies):
        strip = fig.add_subplot(gs[3 * i], sharex=first_ax)
        ax = fig.add_subplot(gs[3 * i + 1], sharex=strip)
        first_ax = first_ax or strip

        _regime_band(strip, s.regimes, alpha=0.9, label=False)
        strip.set_yticks([])
        strip.tick_params(labelbottom=False, length=0)
        for side in strip.spines.values():
            side.set_visible(False)
        strip.set_title(s.name, loc="left", color=INK, fontsize=11)
        strip.text(-0.005, 0.5, "Regime", transform=strip.transAxes, ha="right", va="center",
                   fontsize=8, color=MUTED)

        W = pd.DataFrame({g: s.weights[m].sum(axis=1) for g, m in groups.items()})
        W["Cash"] = s.weights[CASH]
        ax.stackplot(W.index, W.T.values, labels=W.columns, colors=colors,
                     edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(W.index[0], W.index[-1])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        _style(ax)
        if i == 0:                                   # a single legend: weights + regimes
            handles, labels = ax.get_legend_handles_labels()
            handles += [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9) for c in REGIME_COLORS]
            labels += [f"Regime: {r}" for r in REGIME_LABELS]
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.15),
                      frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.97, bottom=0.04)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def bootstrap_boxes(boot, observed, path=None):
    """4) Bootstrap distribution of the Sharpe ratio: one panel per block length, boxes grouped
    and colored by Omega, one box per prior covariance within each group (default boxplot).

    boot     : columns block, prior_cov, omega, sharpe   (results/bootstrap_sharpe.csv)
    observed : columns prior_cov, omega, sharpe          (results/sharpe_observed.csv)"""
    omegas = list(dict.fromkeys(boot["omega"]))
    covs = list(dict.fromkeys(boot["prior_cov"]))
    blocks = sorted(boot["block"].unique())
    obs = observed.set_index(["omega", "prior_cov"])["sharpe"]
    positions = {(o, c): oi * (len(covs) + 1) + ci + 1
                 for oi, o in enumerate(omegas) for ci, c in enumerate(covs)}

    fig, axes = plt.subplots(len(blocks), 1, figsize=(12, 3.6 * len(blocks)), sharex=True,
                             sharey=True, squeeze=False)
    for ax, L in zip(axes[:, 0], blocks):
        g = boot[boot["block"] == L]
        for oi, o in enumerate(omegas):
            pos = [positions[(o, c)] for c in covs]
            data = [g[(g.omega == o) & (g.prior_cov == c)]["sharpe"].values for c in covs]
            bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True,
                            medianprops=dict(color=INK))
            for patch in bp["boxes"]:
                patch.set(facecolor=COLORS[oi], alpha=0.45, edgecolor=COLORS[oi])
            ax.scatter(pos, [obs[(o, c)] for c in covs], color=INK, s=18, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("Sharpe ratio")
        _style(ax, f"Block length: {L} days")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i], alpha=0.45) for i in range(len(omegas))]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK, ms=5))
    axes[0, 0].legend(handles, [f"Ω: {o}" for o in omegas] + ["Observed Sharpe"],
                      loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    axes[-1, 0].set_xticks(list(positions.values()))
    axes[-1, 0].set_xticklabels([c for o, c in positions], fontsize=9)
    axes[-1, 0].set_xlabel("Prior covariance (Σ)", color=MUTED, fontsize=9)
    fig.suptitle("Bootstrap Sharpe ratio of HMM + BL", x=0.01, ha="left", fontsize=12)
    return _save(fig, path)
