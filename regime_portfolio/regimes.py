"""Multivariate HMM (variational Bayes, hmmlearn) with multistart and volatility-ordered states."""
import copy
import warnings

import numpy as np
from hmmlearn.vhmm import VariationalGaussianHMM

warnings.filterwarnings("ignore", module="hmmlearn")


class RegimeModel:
    """Regime model that can be shared by several strategies.

    - Works on daily log-returns in %.
    - Multistart: every fit estimates `n_restarts` candidates (a warm start from the previous
      fit plus fresh initializations) and keeps the one with the highest ELBO. The prior on
      the means is set to 0 for all candidates, so their ELBOs are comparable.
    - States are reordered by decreasing volatility: 0 = crisis, K-1 = calm.
    - Fits and predictions are cached by sample length: strategies with the same calendar
      share a single estimate (the first backtest fits, the others read the cache).
    """

    def __init__(self, n_states=3, seed=123, n_restarts=5, n_iter=500, tol=1e-6):
        self.n_states, self.seed, self.n_restarts = n_states, seed, n_restarts
        self.n_iter, self.tol = n_iter, tol
        self._rng = np.random.default_rng(seed)
        self._fits, self._preds = {}, {}
        self._last = None                     # last model actually estimated (for warm start)

    def _new_hmm(self, random_state, n_features):
        return VariationalGaussianHMM(
            n_components=self.n_states, covariance_type="full", n_iter=self.n_iter, tol=self.tol,
            random_state=random_state, means_prior=np.zeros((self.n_states, n_features)))

    def _estimate(self, X):
        candidates = []
        if self._last is not None:                                  # warm start
            warm = copy.deepcopy(self._last)
            warm.fit(X)
            candidates.append(warm)
        for i in range(self.n_restarts - len(candidates)):         # fresh initializations
            rs = self.seed if (self._last is None and i == 0) else int(self._rng.integers(2**31 - 1))
            h = self._new_hmm(rs, X.shape[1])
            try:
                h.fit(X)
            except (ValueError, np.linalg.LinAlgError):
                continue
            candidates.append(h)
        if not candidates:
            raise RuntimeError("No HMM fit succeeded")
        best = max(candidates, key=lambda h: h.monitor_.history[-1])
        best.init_params = ""                                      # next fits start from here
        self._last = best
        return best

    def fit(self, X):
        n = len(X)
        if n not in self._fits:
            if self._fits and n < max(self._fits):
                raise RuntimeError("Fit on a shorter sample than the last one: the warm start "
                                   "would use future data. Strategies must share the same calendar.")
            hmm = self._estimate(X)
            order = np.argsort(-np.trace(hmm.covars_, axis1=1, axis2=2))
            self._fits[n] = dict(
                hmm=copy.deepcopy(hmm), order=order,
                means=hmm.means_[order],                            # (K, n)
                covars=hmm.covars_[order],                          # (K, n, n)
                transmat=hmm.transmat_[np.ix_(order, order)],       # (K, K)
                n_eff=hmm.predict_proba(X).sum(axis=0)[order])      # effective days per regime
        self.__dict__.update(self._fits[n])
        return self

    def predict_next(self, X):
        """P(s_{t+1} | data up to t): last filtered probability propagated with A."""
        n = len(X)
        if n not in self._preds:
            filtered = self.hmm.predict_proba(X)[-1][self.order]   # at the last date smoothed = filtered
            self._preds[n] = filtered @ self.transmat
        return self._preds[n].copy()

    def mixture_moments(self, p):
        """Mean and covariance of the mixture with weights p (law of total variance):
        total variance = within (average covariance) + between (dispersion of the means)."""
        q = p @ self.means
        within = np.einsum("k,kij->ij", p, self.covars)
        dev = self.means - q
        between = np.einsum("k,ki,kj->ij", p, dev, dev)
        return q, within, between
