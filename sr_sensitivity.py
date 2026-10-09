"""Sensitivity of HMM + BL (statistical Omega) to the Sharpe ratio assumed for the prior.

SR_target in linspace(0.20, 0.90, 10) x 3 prior covariances = 30 backtests. All runs share
one HMM (same calendar), so the regime model is fitted only once. Settings come from main.py.

Outputs:
  figures/sr_sensitivity_paths.png    portfolio value of the 30 runs, one panel per prior covariance
  figures/sr_sensitivity_metrics.png  CAGR, Sharpe, max drawdown and turnover vs SR_target
  results/sr_sensitivity_metrics.csv  metrics of the 30 runs
  results/sr_sensitivity_wealth.csv   portfolio value of the 30 runs (long format)

To redraw the plots from the CSV files only: replot().
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from main import COMMON, CONSTRAINTS, DATA, FIG, N_RESTARTS, OUT, PRIOR_COVS, SEED
from regime_portfolio import BlackLittermanHMM, RegimeModel, load_data
from regime_portfolio.plots import COLORS, _style

SR_TARGETS = np.linspace(0.20, 0.90, 10)
# Sequential blue ramp, light = low SR_target, dark = high SR_target
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def run(n_restarts=N_RESTARTS, sr_targets=SR_TARGETS):
    """Run the 30 backtests and save metrics and portfolio values as CSV."""
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    rets, rf = load_data(DATA)
    model = RegimeModel(seed=SEED, n_restarts=n_restarts)       # shared: fitted only once

    rows, paths = [], []
    for cov in PRIOR_COVS:
        for sr in sr_targets:
            s = BlackLittermanHMM(rets, rf, model=model, omega="statistical", prior_cov=cov,
                                  sr_target=sr, constraints=CONSTRAINTS,
                                  name=f"{cov}, SR {sr:.2f}", **COMMON).run()
            label = round(float(sr), 2)                       # 0.2777... -> 0.28 in CSV and plots
            rows.append({"prior_cov": cov, "sr_target": label, **s.metrics().to_dict()})
            paths.append(s.wealth.rename("wealth").rename_axis("date").reset_index()
                         .assign(prior_cov=cov, sr_target=label))

    metrics = pd.DataFrame(rows)
    wealth = pd.concat(paths, ignore_index=True)
    metrics.to_csv(OUT / "sr_sensitivity_metrics.csv", index=False)
    wealth.to_csv(OUT / "sr_sensitivity_wealth.csv", index=False)
    return metrics, wealth


def plot_paths(wealth, path=None):
    """Portfolio value of every run: one panel per prior covariance, color = SR_target."""
    covs = list(dict.fromkeys(wealth["prior_cov"]))
    srs = sorted(wealth["sr_target"].unique())
    colors = [RAMP[i] for i in np.linspace(0, len(RAMP) - 1, len(srs)).round().astype(int)]
    log_fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    fig, axes = plt.subplots(1, len(covs), figsize=(15, 4.8), sharey=True, squeeze=False)
    for ax, cov in zip(axes[0], covs):
        for color, sr in zip(colors, srs):
            w = wealth[(wealth.prior_cov == cov) & (wealth.sr_target == sr)]
            ax.plot(pd.to_datetime(w["date"]), w["wealth"], color=color, lw=1.2, label=f"{sr:.2f}")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(log_fmt)
        ax.yaxis.set_minor_formatter(log_fmt)
        _style(ax, f"Prior covariance: {cov}")
    axes[0, 0].set_ylabel("Portfolio value (1 = start of test)")
    axes[0, -1].legend(title="SR_target", loc="upper left", bbox_to_anchor=(1.0, 1.0),
                       frameon=False, fontsize=9, title_fontsize=9)
    fig.suptitle("HMM + BL (statistical Ω): portfolio value for different prior Sharpe ratios",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics(metrics, path=None):
    """CAGR, Sharpe, max drawdown and turnover as a function of SR_target."""
    panels = [("CAGR", "CAGR", True), ("Sharpe", "Sharpe ratio", False),
              ("MaxDD", "Max drawdown", True), ("Turnover/yr", "Turnover per year", False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (col, title, pct) in zip(axes.flat, panels):
        for i, (cov, g) in enumerate(metrics.groupby("prior_cov", sort=False)):
            ax.plot(g["sr_target"], g[col], marker="o", ms=4, lw=1.5, color=COLORS[i], label=cov)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _style(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("SR_target (Sharpe ratio assumed for the 1/N prior)")
    axes[0, 1].legend(title="Prior covariance", frameon=False, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def report(metrics, wealth):
    """Print the main metrics and draw the plots."""
    for col, fmt in [("CAGR", "{:.2%}"), ("Sharpe", "{:.2f}"), ("MaxDD", "{:.2%}")]:
        table = metrics.pivot(index="sr_target", columns="prior_cov", values=col)
        print(f"\n{col} by SR_target (rows) and prior covariance (columns):")
        print(table.to_string(float_format=fmt.format, index=True))
    plot_paths(wealth, path=FIG / "sr_sensitivity_paths.png")
    plot_metrics(metrics, path=FIG / "sr_sensitivity_metrics.png")
    plt.show()


def replot():
    """Redraw the plots from the saved CSV files, without rerunning the backtests."""
    report(pd.read_csv(OUT / "sr_sensitivity_metrics.csv"),
           pd.read_csv(OUT / "sr_sensitivity_wealth.csv"))


if __name__ == "__main__":
    report(*run())
