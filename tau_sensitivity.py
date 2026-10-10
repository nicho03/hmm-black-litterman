"""Sensitivity of HMM + BL to the weight of the prior: τ proportional to 1/t or constant.

BL statistical and BL conditional, each with every τ in TAUS. All runs share one HMM,
so the regime model is fitted only once. Settings come from main.py.

Outputs:
  figures/tau_sensitivity_paths.png    portfolio value, one panel per Omega (1/N as reference)
  figures/tau_sensitivity_weights.png  weight of the views in the posterior mean over time
  figures/tau_sensitivity_metrics.png  Sharpe, max drawdown, turnover and mean view weight
  results/tau_sensitivity_metrics.csv  metrics of the runs
  results/tau_sensitivity_series.csv   portfolio value and view weight of the runs (long format)

To redraw the plots from the CSV files only: replot().
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from main import COMMON, CONSTRAINTS, DATA, FIG, MAIN_PRIOR, N_RESTARTS, OUT, SEED
from regime_portfolio import BlackLittermanHMM, FixedWeights, RegimeModel, load_data
from regime_portfolio.plots import COLORS, MUTED, _style

OMEGAS = ["statistical", "conditional"]
# Increasing prior uncertainty. 1/t ranges from 1/1260 to about 1/5200; 0.025 and 0.05 are the
# values common in the literature (Idzorek; He & Litterman)
TAUS = {"τ = 0.0001": 1e-4, "τ = 1/t (default)": None, "τ = 0.025": 0.025, "τ = 0.05": 0.05}


def run(n_restarts=N_RESTARTS):
    """Run the backtests (+ 1/N) and save metrics and time series as CSV."""
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    rets, rf = load_data(DATA)
    model = RegimeModel(seed=SEED, n_restarts=n_restarts)       # shared: fitted only once

    rows, series = [], []
    for omega in OMEGAS:
        for label, tau in TAUS.items():
            s = BlackLittermanHMM(rets, rf, model=model, omega=omega, prior_cov=MAIN_PRIOR, tau=tau,
                                  constraints=CONSTRAINTS, name=f"{omega}, {label.replace('τ', 'tau')}", **COMMON).run()
            vw = pd.Series(s.view_weight)
            rows.append({"omega": omega, "tau": label, **s.metrics().to_dict(), "View weight": vw.mean()})
            series.append(pd.DataFrame({"wealth": s.wealth, "view_weight": vw.reindex(s.wealth.index)})
                          .rename_axis("date").reset_index().assign(omega=omega, tau=label))
    ew = FixedWeights(rets, rf, name="1/N", rebalance=True, **COMMON).run()
    series.append(ew.wealth.rename("wealth").rename_axis("date").reset_index()
                  .assign(omega="benchmark", tau="1/N"))

    metrics, series = pd.DataFrame(rows), pd.concat(series, ignore_index=True)
    metrics.to_csv(OUT / "tau_sensitivity_metrics.csv", index=False)
    series.to_csv(OUT / "tau_sensitivity_series.csv", index=False)
    return metrics, series


def _panels(series, ylabel, col, path, log=False):
    """One panel per Omega, one line per τ."""
    fig, axes = plt.subplots(1, len(OMEGAS), figsize=(14, 4.8), sharey=True)
    ref = series[series.omega == "benchmark"]
    for ax, omega in zip(axes, OMEGAS):
        for i, label in enumerate(TAUS):
            g = series[(series.omega == omega) & (series.tau == label)].dropna(subset=[col])
            ax.plot(pd.to_datetime(g["date"]), g[col], color=COLORS[i], lw=1.4, label=label,
                    drawstyle="steps-post" if col == "view_weight" else "default")
        if col == "wealth":
            ax.plot(pd.to_datetime(ref["date"]), ref["wealth"], color=MUTED, lw=1.2, ls="--", label="1/N")
        if log:
            ax.set_yscale("log")
            fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
        else:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _style(ax, f"HMM + BL ({omega})")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics(metrics, path=None):
    """Grouped bars: one group per Omega, one bar per τ."""
    panels = [("Sharpe", "Sharpe ratio", False), ("MaxDD", "Max drawdown", True),
              ("Turnover/yr", "Turnover per year", False), ("View weight", "Mean weight of the views", True)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    width = 0.8 / len(TAUS)
    for ax, (col, title, pct) in zip(axes, panels):
        for i, label in enumerate(TAUS):
            vals = [metrics[(metrics.omega == o) & (metrics.tau == label)][col].iloc[0] for o in OMEGAS]
            ax.bar([j + (i - (len(TAUS) - 1) / 2) * width for j in range(len(OMEGAS))], vals, width, color=COLORS[i],
                   label=label)
        ax.set_xticks(range(len(OMEGAS)))
        ax.set_xticklabels(OMEGAS)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _style(ax, title)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def report(metrics, series):
    """Print the metrics and draw the plots."""
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover/yr", "Costs/yr", "View weight"]
    fmt = {c: "{:.2%}".format for c in ["CAGR", "Vol", "MaxDD", "Costs/yr", "View weight"]}
    fmt |= {c: "{:.2f}".format for c in ["Sharpe", "Turnover/yr"]}
    table = metrics.assign(tau=metrics["tau"].str.replace("τ", "tau")).set_index(["omega", "tau"])
    print(table[cols].to_string(formatters=fmt))
    _panels(series, "Portfolio value (1 = start of test)", "wealth", FIG / "tau_sensitivity_paths.png", log=True)
    _panels(series, "Weight of the views in the posterior mean", "view_weight",
            FIG / "tau_sensitivity_weights.png")
    plot_metrics(metrics, path=FIG / "tau_sensitivity_metrics.png")
    plt.show()


def replot():
    """Redraw the plots from the saved CSV files, without rerunning the backtests."""
    report(pd.read_csv(OUT / "tau_sensitivity_metrics.csv"), pd.read_csv(OUT / "tau_sensitivity_series.csv"))


if __name__ == "__main__":
    report(*run())
