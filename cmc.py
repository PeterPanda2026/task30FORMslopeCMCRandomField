"""Coupled Markov chain (CMC) toolbox.

Grid convention: column i = 1..Nx runs left -> right,
row j = 1..Nz runs top -> bottom.  Cell (i, j) depends on the left cell
(i-1, j), the upper cell (i, j-1) and - for conditional simulation - the
state of the rightmost column (Nx, j):

    p_{lt,k|q} = p^h_{lk} * p^{h(Nx-i)}_{kq} * p^v_{tk}
                 / sum_f p^h_{lf} * p^{h(Nx-i)}_{fq} * p^v_{tf}

Internally arrays are 0-based: Z[r, j, i] with r = realization index.
"""
import numpy as np

STATES = {1: "clay", 2: "sand", 3: "silt"}


# ----------------------------------------------------------------------------
# Transition matrices
# ----------------------------------------------------------------------------
def row_normalise(T):
    T = np.asarray(T, dtype=float)
    return T / T.sum(axis=1, keepdims=True)


def htpm_from_K(Tv, K):
    """HTPM from vertical transition counts: T^h = T^v with the diagonal
    multiplied by K, then row normalisation."""
    Th = np.asarray(Tv, dtype=float).copy()
    Th[np.diag_indices_from(Th)] *= K
    return row_normalise(Th), Th


def matrix_powers(P, nmax):
    """Return array pw with pw[n] = P^n for n = 0..nmax."""
    m = P.shape[0]
    pw = np.empty((nmax + 1, m, m))
    pw[0] = np.eye(m)
    for n in range(1, nmax + 1):
        pw[n] = pw[n - 1] @ P
    return pw


def conditional_tables(Ph, Pv, Nx):
    """Pre-compute the conditional probability above for every interior column.

    Returns tab[i, l, t, q, k] (0-based column index i = 1..Nx-2) = probability
    that cell (i, j) is in state k given left state l, upper state t and
    right-boundary state q.  Also returns the number of (i,l,t,q) combinations
    whose denominator vanished (these are given a uniform fallback).
    """
    m = Ph.shape[0]
    pw = matrix_powers(Ph, Nx - 1)
    tab = np.zeros((Nx, m, m, m, m))
    n_bad = 0
    for i in range(1, Nx - 1):
        n = (Nx - 1) - i                      # remaining steps to the right column
        # num[l, t, q, k] = Ph[l,k] * pw[n][k,q] * Pv[t,k]
        num = Ph[:, None, None, :] * pw[n].T[None, None, :, :] * Pv[None, :, None, :]
        den = num.sum(axis=-1, keepdims=True)
        bad = den[..., 0] == 0
        n_bad += int(bad.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(den > 0, num / den, 1.0 / m)
        tab[i] = t
    return tab, n_bad


# ----------------------------------------------------------------------------
# Conditional simulation
# ----------------------------------------------------------------------------
def simulate(tab, left, right, top_state, n_sim, rng):
    """Vectorised conditional CMC simulation of n_sim realizations.

    left, right : int arrays (Nz,) with 0-based states of the two conditioning
                  boreholes (columns 0 and Nx-1).
    top_state   : 0-based state imposed on the whole top row (j = 0).
    Returns Z of shape (n_sim, Nz, Nx), dtype int8, 0-based states.
    """
    Nx = tab.shape[0]
    Nz = len(left)
    m = tab.shape[-1]
    Z = np.empty((n_sim, Nz, Nx), dtype=np.int8)
    Z[:, :, 0] = left[None, :]
    Z[:, :, Nx - 1] = right[None, :]
    Z[:, 0, :] = top_state
    Z[:, 0, 0], Z[:, 0, Nx - 1] = left[0], right[0]   # boreholes override the top row at the boundaries
    ar = np.arange(n_sim)
    for j in range(1, Nz):
        q = int(right[j])
        for i in range(1, Nx - 1):
            l = Z[:, j, i - 1]
            t = Z[:, j - 1, i]
            p = tab[i, l, t, q, :]                     # (n_sim, m)
            c = np.cumsum(p, axis=1)
            u = rng.random(n_sim) * c[:, -1]
            k = (u[:, None] >= c).sum(axis=1)
            Z[:, j, i] = np.minimum(k, m - 1)
    return Z


# ----------------------------------------------------------------------------
# Likelihood indicator
# ----------------------------------------------------------------------------
def cell_probabilities(Z, n_states=3):
    """P[k, j, i] = fraction of realizations with cell (i, j) in state k."""
    n_sim = Z.shape[0]
    return np.stack([(Z == k).sum(axis=0) / n_sim for k in range(n_states)])


def log10_likelihood(P, obs_cols, obs_logs, floor=None):
    """log10 P_l = sum over observed cells of log10 P(Z_ij = observed).

    obs_cols : list of 0-based column indices of the observed boreholes
    obs_logs : list of float arrays (Nz,) with 0-based states (NaN = no log)
    floor    : optional lower bound for a cell probability (avoids log 0)
    Returns (log10 P_l, number of observed cells, number of cells with P = 0).
    """
    total, n_cells, n_zero = 0.0, 0, 0
    for ic, log in zip(obs_cols, obs_logs):
        for j, s in enumerate(log):
            if np.isnan(s):
                continue
            p = P[int(s), j, ic]
            n_cells += 1
            if p == 0:
                n_zero += 1
            if floor is not None:
                p = max(p, floor)
            total += np.log10(p) if p > 0 else -np.inf
    return total, n_cells, n_zero
