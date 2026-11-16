"""Vertical and horizontal transition probability matrices (VTPM, HTPM).

State order (rows = from, columns = to): 1 clay, 2 sand, 3 silt.
Step sizes the matrices refer to:
    VTPM : one vertical step   dz = 0.3 m
    HTPM : one horizontal step dx = 0.9 m
The values are given to 3 decimals, so some rows sum to 0.999 or 1.001;
they are row-normalised before use.

If the CMC grid uses a horizontal cell length other than 0.9 m, the HTPM for
that step is P_h(dx) = P_h(0.9 m) ** (dx / 0.9), computed by
eigen-decomposition (htpm_for_step). For dx = 2 m this is used in
cmc_setup.py.
"""
import numpy as np

STATE_NAMES = {1: "clay", 2: "sand", 3: "silt"}
DZ_V = 0.3     # m, step of the VTPM
DX_H = 0.9     # m, step of the HTPM

# VTPM (dz = 0.3 m)
VTPM_RAW = [[0.943, 0.058, 0.000],
            [0.053, 0.931, 0.015],
            [0.000, 0.026, 0.974]]

# HTPM (dx = 0.9 m)
HTPM_RAW = [[0.970, 0.030, 0.000],
            [0.028, 0.964, 0.008],
            [0.000, 0.013, 0.987]]


def row_normalise(P):
    P = np.asarray(P, dtype=float)
    return P / P.sum(axis=1, keepdims=True)


def vtpm():
    """VTPM for dz = 0.3 m (row-normalised)."""
    return row_normalise(VTPM_RAW)


def htpm():
    """HTPM for dx = 0.9 m (row-normalised)."""
    return row_normalise(HTPM_RAW)


def htpm_for_step(dx):
    """HTPM for a horizontal step dx (m): P_h(0.9) ** (dx / 0.9)."""
    P = htpm()
    w, Q = np.linalg.eig(P)
    Pd = (Q @ np.diag(w.astype(complex) ** (dx / DX_H)) @ np.linalg.inv(Q)).real
    Pd = np.clip(Pd, 0.0, None)
    return row_normalise(Pd)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("VTPM (dz = 0.3 m):")
    print(vtpm())
    print("HTPM (dx = 0.9 m):")
    print(htpm())
    print("HTPM converted to dx = 2.0 m:")
    print(htpm_for_step(2.0))
