"""HMM + BL (statistical Ω) with the three estimates of Σ_long, against 1/N and 60/40.

Output: figures/cov_comparison.png  portfolio value, drawdown, calendar-year returns and average
        weights by asset class. Metrics are printed. Settings come from main.py.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from main import COMMON, CONSTRAINTS, DATA, FIG, GROUPS, N_RESTARTS, PRIOR_COVS, SEED
from regime_portfolio import BlackLittermanHMM, FixedWeights, RegimeModel, compare, load_data
from regime_portfolio.base import CASH
from regime_portfolio.plots import CASH_COLOR, COLORS, _style


def main(n_restarts=N_RESTARTS):
    rets, rf = load_data(DATA)
    model = RegimeModel(seed=SEED, n_restarts=n_restarts)
    bl = [BlackLittermanHMM(rets, rf, model=model, omega="statistical", prior_cov=c,
                            constraints=CONSTRAINTS, name=f"BL statistical, {c}", **COMMON).run()
          for c in PRIOR_COVS]
    refs = [FixedWeights(rets, rf, name="1/N", rebalance=True, **COMMON).run(),
            FixedWeights(rets, rf, {"SPY": 0.6, "IEF": 0.4}, name="60/40", **COMMON).run()]
    print()
    compare(bl + refs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    styles = [dict(color=COLORS[i], lw=1.6) for i in range(len(bl))] + \
             [dict(color=COLORS[len(bl) + i], lw=1.2, ls="--") for i in range(len(refs))]

    ax = axes[0, 0]                                            # portfolio value
    for s, st in zip(bl + refs, styles):
        ax.plot(s.wealth.index, s.wealth, label=s.name, **st)
    ax.set_yscale("log")
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Portfolio value (1 = start of test, net of costs)")

    ax = axes[0, 1]                                            # drawdown
    for s, st in zip(bl + refs, styles):
        ax.plot(s.wealth.index, s.wealth / s.wealth.cummax() - 1, **st)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _style(ax, "Drawdown")

    ax = axes[1, 0]                                            # calendar-year returns
    yearly = pd.DataFrame({s.name: (1 + s.returns).groupby(s.returns.index.year).prod() - 1
                           for s in bl + refs[:1]})
    width = 0.8 / yearly.shape[1]
    x = np.arange(len(yearly))
    for i, col in enumerate(yearly):
        ax.bar(x + (i - (yearly.shape[1] - 1) / 2) * width, yearly[col], width,
               color=COLORS[i], label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index, rotation=90, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _style(ax, "Calendar-year returns")

    ax = axes[1, 1]                                            # average weights by asset class
    W = pd.DataFrame({s.name.split(", ")[1]: {**{g: s.weights[m].sum(axis=1).mean() for g, m in GROUPS.items()},
                                               "Cash": s.weights[CASH].mean()} for s in bl}).T
    print("\nAverage weights by asset class:")
    print(W.to_string(float_format="{:.1%}".format))
    W.plot.barh(stacked=True, ax=ax, color=COLORS[:len(GROUPS)] + [CASH_COLOR], width=0.6, legend=False)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    _style(ax, "Average weights by asset class (BL statistical)")

    fig.tight_layout()
    fig.savefig(FIG / "cov_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
