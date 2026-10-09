"""Walk-forward backtest of HMM + Black-Litterman against passive benchmarks.

Outputs:
  figures/performance.png       1) portfolio value, background = regime predicted by the HMM
  results/metrics.csv           2) metrics table (also printed)
  figures/bl_weights.png        3) Black-Litterman weights by asset class, with predicted regime
  figures/bootstrap_sharpe.png  4) bootstrap box plots of the Sharpe ratio, grouped by Omega
  results/bootstrap_sharpe.csv  5) box plot data (+ sharpe_observed.csv, bootstrap_pairs.csv)

To redraw plot 4 from the CSV files only: replot_bootstrap().
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from regime_portfolio import (BlackLittermanHMM, FixedWeights, MarkowitzHMM, RegimeModel,
                              bootstrap_sharpe, compare, load_data, pairwise_tests, plots, sharpe)
from regime_portfolio.constraints import group_bounds, max_weight

ROOT = Path(__file__).parent
DATA, FIG, OUT = ROOT / "data", ROOT / "figures", ROOT / "results"

GROUPS = {                                   # asset classes (USD cash is added separately)
    "Equity": ["SPY", "EFA", "EEM"],
    "Treasury": ["TLT", "IEF", "TIP"],
    "Corporate": ["LQD"],
    "Real assets": ["GLD", "DBC"],
}
COMMON = dict(n_train=252 * 5, rebalance_every=20, risk_aversion=0.05, spread_bps=10)
CONSTRAINTS = [max_weight(0.35), group_bounds(GROUPS["Equity"], 0.00, 0.70)]
OMEGAS = ["statistical", "idzorek", "conditional"]
PRIOR_COVS = ["sample", "ledoit_wolf", "mcd"]
MAIN_PRIOR = "sample"                        # prior covariance used in plots 1 and 3
SEED, N_RESTARTS = 123, 5                    # HMM: seed and multistart at every fit
BLOCKS, N_BOOT = (20, 60, 120), 2000         # bootstrap: block lengths (days), replicates


def main(n_restarts=N_RESTARTS, n_boot=N_BOOT):
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    rets, rf = load_data(DATA)

    # One HMM for all regime strategies: same calendar, hence the same estimate
    model = RegimeModel(seed=SEED, n_restarts=n_restarts)
    hmm_kw = dict(model=model, constraints=CONSTRAINTS, **COMMON)

    benchmarks = [
        FixedWeights(rets, rf, {"SPY": 1.0}, name="S&P 500 (SPY)", **COMMON),
        FixedWeights(rets, rf, {"SPY": 0.6, "IEF": 0.4}, name="60/40", **COMMON),
        FixedWeights(rets, rf, name="1/N",rebalance = True, **COMMON),
    ]
    markowitz = MarkowitzHMM(rets, rf, **hmm_kw)
    bl = {(o, c): BlackLittermanHMM(rets, rf, omega=o, prior_cov=c, **hmm_kw)
          for c in PRIOR_COVS for o in OMEGAS}
    for s in [*benchmarks, markowitz, *bl.values()]:
        s.run()

    # 1) performance: benchmarks, Markowitz and BL with the three Omegas
    main_bl = [bl[(o, MAIN_PRIOR)] for o in OMEGAS]
    plots.performance([*benchmarks, markowitz, *main_bl],
                      regimes=bl[("statistical", MAIN_PRIOR)].regimes, path=FIG / "performance.png")

    # 2) metrics table
    print()
    compare([*benchmarks, markowitz, *bl.values()]).to_csv(OUT / "metrics.csv")

    # 3) BL portfolio weights
    plots.weights_by_group(main_bl, GROUPS, path=FIG / "bl_weights.png")

    # 4-5) block bootstrap of the Sharpe ratio of the 9 BL configurations
    excess = pd.DataFrame({s.name: s.excess_returns for s in bl.values()})
    labels = pd.DataFrame([{"strategy": s.name, "omega": o, "prior_cov": c} for (o, c), s in bl.items()])
    observed = labels.assign(sharpe=[sharpe(excess[n].to_numpy()) for n in labels.strategy])
    boot = bootstrap_sharpe(excess, BLOCKS, B=n_boot).merge(labels, on="strategy")
    pairs = pairwise_tests(boot, observed.set_index("strategy")["sharpe"])

    boot.to_csv(OUT / "bootstrap_sharpe.csv", index=False)
    observed.to_csv(OUT / "sharpe_observed.csv", index=False)
    pairs.to_csv(OUT / "bootstrap_pairs.csv", index=False)
    print("\nPairs of BL configurations with significantly different Sharpe (Holm, 5%):")
    print(pairs.groupby("block")["significant"].agg(["sum", "count"])
          .rename(columns={"sum": "significant", "count": "pairs"}).to_string())
    plots.bootstrap_boxes(boot, observed, path=FIG / "bootstrap_sharpe.png")

    plt.show()


def replot_bootstrap():
    """Redraw plot 4 from the saved CSV files, without rerunning the backtest."""
    plots.bootstrap_boxes(pd.read_csv(OUT / "bootstrap_sharpe.csv"),
                          pd.read_csv(OUT / "sharpe_observed.csv"), path=FIG / "bootstrap_sharpe.png")
    plt.show()


if __name__ == "__main__":
    main()
