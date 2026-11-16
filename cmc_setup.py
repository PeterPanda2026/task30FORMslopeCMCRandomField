"""Coupled Markov chain (CMC) inputs and the link between the CMC grid and
the slope model.

Site data given to the analysis
-------------------------------
boreholes_slope.csv  - five boreholes drilled from the slope ground surface:
    BH1 (x = 0 m) from the crest level,  collar elevation 28.2 m
    BH2 (x = 20 m) from the slope face,  collar elevation 26.2 m
    BH3 (x = 40 m) from the slope face,  collar elevation 18.2 m
    BH4 (x = 50 m) and BH5 (x = 70 m) from the toe level, collar elevation 14.7 m
    sampled every 0.3 m at elevations 28.2 - 0.3 k (so the first sample of
    BH2 / BH3 is 0.1 / 0.2 m below the collar); state 1 = clay, 2 = sand, 3 = silt.
    depth_m is measured from each borehole's collar; elevation_m is absolute.
    BH4 has no log at elevation 0.9 and 0.6 m (below the model base, 1.2 m).
ground_surface_soil.csv - soil type mapped along the whole ground surface:
    crest (every 2 m), slope face (every 0.3 m of height), toe level (every 2 m).

Transition probability matrices (transition_matrices.py)
--------------------------------------------------------
    VTPM for dz = 0.3 m, HTPM for dx = 0.9 m.
With the 2 m CMC cells used here, HTPM(2 m) = HTPM(0.9 m) ** (2 / 0.9).

CMC grid
--------
    cell height dz = 0.3 m, cell length dx = 2.0 m
    rows    j = 0..92, cell-centre elevation = 28.2 - 0.3 j
    columns i = 0..35, cell-centre x         = 2.0 i
    boreholes occupy columns 0, 10, 20, 25, 35
    only cells whose centre lies inside the slope are soil (see slope_mask)

Monte Carlo settings
--------------------
    N_sim = 500 realizations for the mean and standard deviation of FS
    N_sim = 10,000 realizations for the probability of failure
    Pf = P(FS < 1)
    each realization combines one CMC soil layout with one sample of the
    random soil parameters (clay c, sand phi; see soil_properties.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import slope_geometry as geo
import transition_matrices as tpm

HERE = Path(__file__).resolve().parent
BOREHOLE_CSV = HERE / "boreholes_slope.csv"
SURFACE_CSV = HERE / "ground_surface_soil.csv"

STATE_NAMES = {1: "clay", 2: "sand", 3: "silt"}
N_STATES = 3
DZ = 0.3                 # m, CMC cell height (= VTPM step)
DX = 2.0                 # m, CMC cell length
TOP_ELEV = 28.2          # m, centre of the top row of cells (crest level)
NZ = 93                  # rows: elevation 28.2 ... 0.6 m

N_SIM_MOMENTS = 500      # realizations for mean / std of FS
N_SIM_PF = 10_000        # realizations for Pf
FS_FAILURE = 1.0         # failure when FS < 1


# ----------------------------------------------------------------------------
# Site data
# ----------------------------------------------------------------------------
def row_of(elev):
    return np.rint((TOP_ELEV - np.asarray(elev, float)) / DZ).astype(int)


def load_boreholes(csv=BOREHOLE_CSV):
    """Return (df, logs, xs).
    logs[b] : float array (NZ,) on the CMC rows, state codes 1..3,
              NaN above the collar and where there is no log
    xs[b]   : x position (m)"""
    df = pd.read_csv(csv)
    logs, xs = {}, {}
    for b, g in df.groupby("borehole_id"):
        log = np.full(NZ, np.nan)
        log[row_of(g["elevation_m"])] = g["state"].to_numpy(dtype=float)
        logs[b], xs[b] = log, float(g["x_m"].iloc[0])
    return df, logs, xs


def load_surface(csv=SURFACE_CSV):
    """Ground-surface soil: DataFrame with segment, x_m, elevation_m, state, state_name."""
    return pd.read_csv(csv)


def transition_matrices(dx=DX):
    """Return (Pv, Ph): VTPM (dz = 0.3 m) and HTPM for step dx (index 0 = clay)."""
    return tpm.vtpm(), tpm.htpm_for_step(dx)


# ----------------------------------------------------------------------------
# CMC grid <-> slope model
# ----------------------------------------------------------------------------
def grid(xs):
    """Cell-centre coordinates: (x (Nx,), elev (NZ,), borehole columns)."""
    x0 = min(xs.values())
    Nx = int(round((max(xs.values()) - x0) / DX)) + 1
    x = x0 + DX * np.arange(Nx)
    elev = TOP_ELEV - DZ * np.arange(NZ)
    cols = {b: int(round((xs[b] - x0) / DX)) for b in xs}
    return x, elev, cols


def slope_mask(x, elev):
    """(NZ, Nx) bool: CMC cells whose centre lies inside the slope body."""
    X, E = np.meshgrid(x, elev)
    return geo.inside_slope(X, E + 1e-9)


def surface_cells(x, elev, surface=None):
    """Top soil cell of every CMC column and its mapped surface soil.
    Returns DataFrame: column i, x, row j, elevation, state (from the
    nearest ground-surface observation)."""
    surface = load_surface() if surface is None else surface
    m = slope_mask(x, elev)
    rows = []
    for i, xi in enumerate(x):
        j = int(np.argmax(m[:, i]))
        d = (surface["x_m"] - xi) ** 2 + (surface["elevation_m"] - elev[j]) ** 2
        rows.append((i, xi, j, elev[j], int(surface["state"].iloc[int(np.argmin(d))])))
    return pd.DataFrame(rows, columns=["col", "x_m", "row", "elevation_m", "state"])


def realization_to_slope(Z, x, elev, zero_based=True):
    """Map one CMC realization (NZ, Nx) to soil codes in the slope.
    Returns int array (NZ, Nx): 1/2/3 inside the slope, 0 outside.
    zero_based=True when Z uses 0/1/2 (as cmc.py does)."""
    S = np.asarray(Z, dtype=int) + (1 if zero_based else 0)
    return np.where(slope_mask(x, elev), S, 0)


def state_at(S, x, elev, xq, eq):
    """Soil code at arbitrary points (xq, eq), e.g. points on the base of slices:
    nearest CMC cell centre. S is (NZ, Nx) from realization_to_slope."""
    i = np.clip(np.rint((np.asarray(xq) - x[0]) / DX).astype(int), 0, len(x) - 1)
    j = np.clip(row_of(eq), 0, len(elev) - 1)
    return S[j, i]


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    df, logs, xs = load_boreholes()
    x, elev, cols = grid(xs)
    Pv, Ph = transition_matrices()
    print(f"CMC grid: {NZ} rows x {len(x)} columns, dz = {DZ} m, dx = {DX} m")
    print("Borehole columns:", cols)
    collars = df.groupby("borehole_id")["collar_elev_m"].first()
    for b in logs:
        top = np.argmax(~np.isnan(logs[b]))
        print(f"  {b}: collar at {collars[b]:.1f} m, first logged cell at {elev[top]:.1f} m, "
              f"{np.sum(~np.isnan(logs[b]))} logged cells")
    print("\nVTPM (dz = 0.3 m):\n", Pv)
    print("\nHTPM (converted to dx = 2 m):\n", Ph)
    m = slope_mask(x, elev)
    print(f"\nCMC cells inside the slope: {m.sum()} of {m.size}")
    sc = surface_cells(x, elev)
    print("\nTop soil cell of each column (state):")
    print(" ".join(f"{int(r.x_m)}:{STATE_NAMES[r.state][:2]}" for r in sc.itertuples()))
