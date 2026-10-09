"""Base class: walk-forward backtest with costs, mean-variance optimizer and metrics."""
from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np
import pandas as pd

CASH = "USD"


class PortfolioStrategy(ABC):
    """Allocation strategy with a walk-forward backtest.

    Every `rebalance_every` days the strategy picks new weights with `allocate`, using only
    data up to the previous day; between rebalances the weights drift with prices.
    The universe is the assets in `rets` plus USD cash, which earns the T-bill rate.

    rets            daily log-returns (decimal) of the risky assets
    rf              daily simple return of cash, aligned with rets
    n_train         days of history before the first rebalance
    rebalance_every days between rebalances (= horizon H of the optimization)
    risk_aversion   δ in daily % units (0.05 ≈ δ = 5 in decimal units)
    spread_bps      proportional transaction cost, always charged in the backtest
    include_costs   if True the optimizer penalizes turnover with the same cost
    constraints     extra constraints (see constraints.py)
    """

    def __init__(self, rets, rf, name=None, n_train=252 * 5, rebalance_every=20,
                 risk_aversion=0.05, spread_bps=10.0, include_costs=True, constraints=()):
        self.name = name or type(self).__name__
        self.risky = list(rets.columns)
        self.assets = self.risky + [CASH]
        self.n = len(self.assets)
        self.X = rets.to_numpy() * 100                          # log-returns in %, for the models
        self.rf = rf.reindex(rets.index).ffill().fillna(0.0)
        self.R = np.expm1(rets.iloc[n_train:])                  # simple returns of the test period
        self.R[CASH] = self.rf.iloc[n_train:]
        self.n_train, self.rebalance_every = n_train, rebalance_every
        self.risk_aversion, self.spread_bps = risk_aversion, spread_bps
        self.include_costs, self.constraints = include_costs, list(constraints)
        self.w_ref = np.ones(len(self.risky)) / len(self.risky)   # 1/N on the risky assets
        self.w_start = np.r_[self.w_ref, 0.0]                     # initial portfolio

    @abstractmethod
    def allocate(self, t, w_cur):
        """Target weights (cash included, summing to 1) decided with data X[:t]."""

    def regime(self):
        """Regime predicted at the last rebalance (0 = crisis), None if there is no model."""
        return None

    # ------------------------------------------------------------------ optimizer
    def optimize(self, mu, Sigma, w_cur, t):
        """max_w  H·μ'w − (δ/2)·H·w'Σw − c·‖w − w_cur‖₁   s.t. Σw = 1, w ≥ 0 and extra constraints.

        mu, Sigma refer to the risky assets (daily %); cash is added with a return equal to
        the current T-bill rate and zero variance."""
        H = self.rebalance_every
        mu = np.r_[mu, 100 * self.rf.iloc[t]]
        Sigma = np.pad((Sigma + Sigma.T) / 2, ((0, 1), (0, 1)))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(self.n))     # w'Σw = ||L'w||²

        w = cp.Variable(self.n)
        objective = H * mu @ w - self.risk_aversion / 2 * H * cp.sum_squares(L.T @ w)
        if self.include_costs:
            objective -= self.spread_bps / 100 * cp.norm1(w - w_cur)   # cost in %
        cons = [cp.sum(w) == 1, w >= 0] + [c for f in self.constraints for c in f(w, self.assets)]
        problem = cp.Problem(cp.Maximize(objective), cons)
        problem.solve(solver=cp.CLARABEL)
        if problem.status != "optimal":
            self.failures += 1
            return w_cur                                           # on failure, do not trade
        w_opt = np.clip(w.value, 0, None)                          # remove numerical -1e-9
        return w_opt / w_opt.sum()

    # ------------------------------------------------------------------- backtest
    def run(self):
        R = self.R.to_numpy()
        T_max, dT = len(R), self.rebalance_every
        ret, cost, turnover = np.zeros(T_max), np.zeros(T_max), np.zeros(T_max)
        weights, regimes = np.zeros((T_max, self.n)), np.full(T_max, np.nan)
        w_cur, self.failures = self.w_start.copy(), 0

        for T in range(0, T_max, dT):
            w_new = self.allocate(self.n_train + T, w_cur)         # data up to the previous day
            turnover[T] = np.abs(w_new - w_cur).sum()
            cost[T] = self.spread_bps / 1e4 * turnover[T]          # fraction of wealth
            reg = self.regime()
            regimes[T:T + dT] = np.nan if reg is None else reg
            w = w_new
            for i, r in enumerate(R[T:T + dT]):
                weights[T + i], ret[T + i] = w, w @ r
                w = w * (1 + r) / (w @ (1 + r))                    # weights drift with prices
            ret[T] -= cost[T]
            w_cur = w

        idx = self.R.index
        self.returns = pd.Series(ret, idx, name=self.name)
        self.weights = pd.DataFrame(weights, idx, self.assets)
        self.costs, self.turnover = pd.Series(cost, idx), pd.Series(turnover, idx)
        self.regimes = pd.Series(regimes, idx)
        print(f"{self.name}: done" + (f" ({self.failures} failed optimizations)" if self.failures else ""))
        return self

    # -------------------------------------------------------------------- metrics
    @property
    def wealth(self):
        return (1 + self.returns).cumprod()

    @property
    def excess_returns(self):
        return self.returns - self.rf.loc[self.returns.index]

    def metrics(self):
        r, wealth = self.returns, self.wealth
        years = len(r) / 252
        cagr = wealth.iloc[-1] ** (1 / years) - 1
        max_dd = (wealth / wealth.cummax() - 1).min()
        ex = self.excess_returns
        q = np.quantile(r, 0.01)                                   # historical daily VaR / ES
        return pd.Series({
            "CAGR": cagr,
            "Vol": r.std() * np.sqrt(252),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(252),
            "MaxDD": max_dd,
            "Calmar": cagr / abs(max_dd),
            "VaR 99%": -q,
            "ES 99%": -r[r <= q].mean(),
            "Turnover/yr": self.turnover.sum() / years,
            "Costs/yr": self.costs.sum() / years,
        }, name=self.name)


def compare(strategies):
    """Metrics table, one row per strategy (printed and returned)."""
    table = pd.DataFrame([s.metrics() for s in strategies])
    fmt = {c: "{:.2%}".format for c in ["CAGR", "Vol", "MaxDD", "VaR 99%", "ES 99%", "Costs/yr"]}
    fmt |= {c: "{:.2f}".format for c in ["Sharpe", "Calmar", "Turnover/yr"]}
    print(table.to_string(formatters=fmt))
    return table
