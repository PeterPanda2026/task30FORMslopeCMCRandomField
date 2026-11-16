"""Random field simulation of the uncertain soil parameters on the CMC grid.

soil_properties.py lists two random soil parameters, the clay cohesion c and
the sand friction angle phi (both lognormal). This script models each of them
as a stationary, anisotropic lognormal RANDOM FIELD instead of a single random
variable, on the same grid as the coupled Markov chain (cmc_setup.py):

    cells dx = 2.0 m (36 columns, x = 0 ... 70 m), dz = 0.3 m (93 rows,
    elevation 28.2 ... 0.6 m); only cells inside the slope are soil.

Model
    ln X(x, z) = mu_ln + sigma_ln * G(x, z),   G = standard Gaussian field
    mu_ln, sigma_ln from the mean and COV in soil_properties.py.
    Correlation of G is separable:  rho(tx, tz) = rho_h(tx) * rho_v(tz),
    each written in terms of a scale of fluctuation delta (Vanmarcke):
        markov               exp(-2 t / delta)
        gaussian             exp(-pi (t / delta)^2)
        second_order_markov  (1 + 4 t / delta) exp(-4 t / delta)
        cosine_exponential   exp(-t / a) cos(t / a),  a = delta
    Default scales: delta_h = 20 m, delta_v = 2 m (--delta-h, --delta-v).
    The clay c field and the sand phi field are independent of each other and
    of the soil layout, as in soil_properties.py.

Simulation methods (--method)
    cmd  covariance matrix decomposition. For a separable correlation on a
         regular grid C = C_v (x) C_h (Kronecker product), so a square root is
         A_v (x) A_h with A A^T = C (from eigen-decomposition, which is also
         stable for the nearly singular Gaussian ACF). One realization is
         G = A_v Xi A_h^T, Xi = (Nz, Nx) independent N(0, 1). Exact on the grid.
    kl   discrete Karhunen-Loeve expansion: eigenpairs of C_v and C_h give the
         2-D eigenpairs lambda_v * lambda_h, phi_v * phi_h; the largest are
         kept until --energy of the total variance is reached; the truncated
         field is rescaled to unit variance at every cell (--no-rescale to
         keep the truncated variance).

Combining with a soil layout
    property_maps(S, c_field, phi_field) puts the clay c field in clay cells,
    the sand phi field in sand cells and the fixed values elsewhere, where S
    is a soil layout from cmc_setup.realization_to_slope (1/2/3, 0 outside).

Outputs (saved next to this script)
    random_field_acf.png           the four correlation functions
    random_field_realizations.png  clay c and sand phi fields + one CMC layout
                                   with the fields assigned to its soils
    random_field_check.png         simulated vs target correlation and moments
                                   (only with --verify)

Usage
    python random_field.py                              # cmd, markov, 3 realizations
    python random_field.py --acf gaussian --method kl --energy 0.95
    python random_field.py --delta-h 10 --delta-v 1 --verify --n-check 2000

    from random_field import SeparableRandomField, simulate_soil_fields
    rf = SeparableRandomField(x, elev, "markov", 20.0, 2.0)
    fields = simulate_soil_fields(rf, 10, np.random.default_rng(1))
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cmc
import cmc_setup as cs
import slope_geometry as geo
import soil_properties as sp

HERE = Path(__file__).resolve().parent
ACFS = ["markov", "gaussian", "second_order_markov", "cosine_exponential"]
RANDOM_PARAMS = {"clay": "c_kPa", "sand": "phi_deg"}      # soil -> its random parameter
SOIL_CODE = {name: code for code, name in sp.STATE_NAMES.items()}


# ============================================================================
# Correlation functions
# ============================================================================
def correlation(acf, tau, delta):
    """Correlation at lag tau (m) for scale of fluctuation delta (m)."""
    t = np.abs(np.asarray(tau, dtype=float))
    if delta <= 0:
        raise ValueError("the scale of fluctuation must be positive")
    if acf == "markov":
        return np.exp(-2.0 * t / delta)
    if acf == "gaussian":
        return np.exp(-np.pi * (t / delta) ** 2)
    if acf == "second_order_markov":
        return (1.0 + 4.0 * t / delta) * np.exp(-4.0 * t / delta)
    if acf == "cosine_exponential":
        return np.exp(-t / delta) * np.cos(t / delta)
    raise ValueError(f"unknown ACF '{acf}'; choose from {ACFS}")


def scale_of_fluctuation(acf, delta, t_max=None, n=20001):
    """Numerical check: delta = 2 * integral_0^inf rho(t) dt."""
    t_max = 40.0 * delta if t_max is None else t_max
    t = np.linspace(0.0, t_max, n)
    r = correlation(acf, t, delta)
    return 2.0 * float(np.sum(0.5 * (r[1:] + r[:-1]) * np.diff(t)))


def correlation_matrix(coords, acf, delta):
    """1-D correlation matrix between grid coordinates."""
    c = np.asarray(coords, dtype=float)
    return correlation(acf, c[:, None] - c[None, :], delta)


def _eig_sorted(C):
    """Eigenvalues (descending, negatives from round-off set to 0) and vectors."""
    lam, vec = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return np.clip(lam[order], 0.0, None), vec[:, order]


# ============================================================================
# Separable Gaussian random field on the CMC grid
# ============================================================================
class SeparableRandomField:
    """Standard Gaussian field on the grid (elev rows x x columns)."""

    def __init__(self, x, elev, acf="markov", delta_h=20.0, delta_v=2.0,
                 method="cmd", energy=0.95, rescale=True, acf_v=None):
        self.x, self.elev = np.asarray(x, float), np.asarray(elev, float)
        self.acf_h, self.acf_v = acf, acf if acf_v is None else acf_v
        self.delta_h, self.delta_v = float(delta_h), float(delta_v)
        self.method, self.energy, self.rescale = method, float(energy), rescale
        self.lam_h, self.vec_h = _eig_sorted(correlation_matrix(self.x, self.acf_h, self.delta_h))
        self.lam_v, self.vec_v = _eig_sorted(correlation_matrix(self.elev, self.acf_v, self.delta_v))
        if method == "cmd":
            self.A_h = self.vec_h * np.sqrt(self.lam_h)          # A A^T = C_h
            self.A_v = self.vec_v * np.sqrt(self.lam_v)
            self.n_terms = len(self.x) * len(self.elev)
        elif method == "kl":
            self._truncate()
        else:
            raise ValueError("method must be 'cmd' or 'kl'")

    def _truncate(self):
        """Keep the largest 2-D eigenvalues lambda_v[j] * lambda_h[i] up to the energy."""
        if not 0 < self.energy <= 1:
            raise ValueError("energy must be in (0, 1]")
        lam2 = np.outer(self.lam_v, self.lam_h).ravel()
        order = np.argsort(lam2)[::-1]
        share = np.cumsum(lam2[order]) / lam2.sum()
        k = int(np.searchsorted(share, self.energy - 1e-12)) + 1
        keep = order[:k]
        self.kl_v, self.kl_h = np.unravel_index(keep, (len(self.lam_v), len(self.lam_h)))
        self.kl_lam = lam2[keep]
        self.n_terms, self.energy_kept = k, float(share[k - 1])
        # point variance of the truncated field, used to rescale to unit variance
        self.kl_var = np.einsum("k,jk,ik->ji", self.kl_lam,
                                self.vec_v[:, self.kl_v] ** 2, self.vec_h[:, self.kl_h] ** 2)

    def simulate(self, n=1, rng=None):
        """n realizations, array (n, Nz, Nx) of standard Gaussian values."""
        rng = np.random.default_rng() if rng is None else rng
        nz, nx = len(self.elev), len(self.x)
        if self.method == "cmd":
            xi = rng.standard_normal((n, nz, nx))
            return np.einsum("ja,nab,ib->nji", self.A_v, xi, self.A_h, optimize=True)
        xi = rng.standard_normal((n, self.n_terms))
        coef = xi * np.sqrt(self.kl_lam)
        g = np.einsum("nk,jk,ik->nji", coef, self.vec_v[:, self.kl_v], self.vec_h[:, self.kl_h],
                      optimize=True)
        return g / np.sqrt(self.kl_var) if self.rescale else g

    def target_correlation(self, direction, lags):
        if direction == "h":
            return correlation(self.acf_h, lags, self.delta_h)
        return correlation(self.acf_v, lags, self.delta_v)

    def describe(self):
        txt = (f"{self.method.upper()} field, ACF {self.acf_h} (h) / {self.acf_v} (v), "
               f"delta_h = {self.delta_h:g} m, delta_v = {self.delta_v:g} m, "
               f"grid {len(self.elev)} x {len(self.x)}")
        if self.method == "kl":
            txt += f", {self.n_terms} KL terms ({self.energy_kept:.1%} of the variance)"
        return txt


# ============================================================================
# Soil parameter fields
# ============================================================================
def to_lognormal(g, mean, cov):
    """Standard Gaussian field -> lognormal field with the given mean and COV."""
    mu, s = sp.lognormal_params(mean, cov)
    return np.exp(mu + s * g)


def simulate_soil_fields(rf, n, rng):
    """Independent fields of every random soil parameter.
    Returns {soil name: (parameter key, array (n, Nz, Nx))}."""
    out = {}
    for soil, key in RANDOM_PARAMS.items():
        v = sp.SOIL_PROPERTIES[soil][key]
        if not sp.is_random(v):
            continue
        if v["dist"] != "lognormal":
            raise ValueError(f"{soil} {key}: only lognormal parameters are simulated as fields")
        out[soil] = (key, to_lognormal(rf.simulate(n, rng), v["mean"], v["cov"]))
    return out


def property_maps(S, fields):
    """Cohesion and friction angle maps (Nz, Nx) for one soil layout S (1/2/3,
    0 outside the slope) and one realization of every field, given as
    {soil name: (parameter key, array (Nz, Nx))}. NaN outside the slope."""
    S = np.asarray(S)
    maps = {"c_kPa": np.full(S.shape, np.nan), "phi_deg": np.full(S.shape, np.nan)}
    for code, name in sp.STATE_NAMES.items():
        cells = S == code
        for key in maps:
            if name in fields and fields[name][0] == key:
                maps[key][cells] = fields[name][1][cells]
            else:
                maps[key][cells] = sp.mean_value(sp.SOIL_PROPERTIES[name][key])
    return maps


def cmc_layout(rng):
    """One CMC soil layout conditioned on the outer boreholes BH1 and BH5
    (rows above a collar take the first logged state below it)."""
    _, logs, xs = cs.load_boreholes()
    x, elev, cols = cs.grid(xs)
    Pv, Ph = cs.transition_matrices()
    tab, _ = cmc.conditional_tables(Ph, Pv, len(x))

    def filled(log):
        log = log.copy()
        first = int(np.argmax(~np.isnan(log)))
        log[:first] = log[first]
        for j in range(1, len(log)):                     # gaps below the base
            if np.isnan(log[j]):
                log[j] = log[j - 1]
        return log.astype(int) - 1

    left = filled(logs[min(cols, key=cols.get)])
    right = filled(logs[max(cols, key=cols.get)])
    top = int(cs.surface_cells(x, elev)["state"].iloc[0]) - 1
    Z = cmc.simulate(tab, left, right, top, 1, rng)[0]
    return cs.realization_to_slope(Z, x, elev), x, elev


# ============================================================================
# Verification
# ============================================================================
def empirical_correlation(g, axis, max_lag):
    """Sample correlation of standardised fields along one grid axis (1 = rows,
    2 = columns), averaged over all positions; lags 0..max_lag cells."""
    g = (g - g.mean(axis=0)) / g.std(axis=0)
    n = g.shape[axis]
    r = []
    for k in range(max_lag + 1):
        a = np.take(g, np.arange(0, n - k), axis=axis)
        b = np.take(g, np.arange(k, n), axis=axis)
        r.append(float(np.mean(a * b)))
    return np.array(r)


def verify(rf, n, rng, path):
    print(f"\nVerification with {n} realizations ({rf.describe()})")
    for acf in ACFS:
        print(f"  {acf:<20} delta from 2 * integral of rho: {scale_of_fluctuation(acf, 2.0):.4f} (target 2)")
    g = rf.simulate(n, rng)
    print(f"  Gaussian field: mean {g.mean():+.4f} (0), point variance {g.var(axis=0).mean():.4f} (1)")
    lag_h, lag_v = 10, 15
    rh = empirical_correlation(g, 2, lag_h)
    rv = empirical_correlation(g, 1, lag_v)
    th = rf.target_correlation("h", cs.DX * np.arange(lag_h + 1))
    tv = rf.target_correlation("v", cs.DZ * np.arange(lag_v + 1))
    print(f"  max |simulated - target| correlation: horizontal {np.abs(rh - th).max():.3f}, "
          f"vertical {np.abs(rv - tv).max():.3f}")
    fields = simulate_soil_fields(rf, n, rng)
    for soil, (key, f) in fields.items():
        v = sp.SOIL_PROPERTIES[soil][key]
        print(f"  {soil} {key}: mean {f.mean():.3f} ({v['mean']:g}), "
              f"COV {(f.std(axis=0) / f.mean(axis=0)).mean():.3f} ({v['cov']:g})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lags, sim, tgt, lab in [(axes[0], cs.DX * np.arange(lag_h + 1), rh, th, "horizontal"),
                                    (axes[1], cs.DZ * np.arange(lag_v + 1), rv, tv, "vertical")]:
        ax.plot(lags, tgt, "k-", label="target")
        ax.plot(lags, sim, "o", ms=4, label=f"simulated ({n} fields)")
        ax.set_xlabel(f"{lab} lag (m)")
        ax.set_ylabel("correlation")
        ax.set_title(f"{lab} correlation", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(rf.describe(), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


# ============================================================================
# Figures
# ============================================================================
def plot_acfs(delta_h, delta_v, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, delta, lab in [(axes[0], delta_h, "horizontal"), (axes[1], delta_v, "vertical")]:
        t = np.linspace(0, 2.5 * delta, 400)
        for acf in ACFS:
            ax.plot(t, correlation(acf, t, delta), label=acf)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(f"{lab} lag (m)")
        ax.set_ylabel("correlation")
        ax.set_title(f"{lab}: scale of fluctuation {delta:g} m", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def _show(ax, x, elev, values, mask, title, cmap, label, fig):
    v = np.where(mask, values, np.nan)
    extent = [x[0] - cs.DX / 2, x[-1] + cs.DX / 2, elev[-1] - cs.DZ / 2, elev[0] + cs.DZ / 2]
    im = ax.imshow(v, extent=extent, aspect="auto", cmap=cmap, origin="upper",
                   interpolation="nearest")
    ax.plot(*geo.SLOPE_POLYGON[[0, 1, 2, 3, 4, 5, 0]].T, "k-", lw=0.8)
    for b, bx in geo.BOREHOLE_X.items():
        ax.plot([bx, bx], [geo.BASE_ELEV, float(geo.ground_surface(bx))], "w-", lw=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)


def plot_realizations(rf, n_show, rng, path):
    S, x, elev = cmc_layout(rng)
    mask = S > 0
    fields = simulate_soil_fields(rf, n_show, rng)
    c_key, c_f = fields["clay"]
    p_key, p_f = fields["sand"]
    fig, axes = plt.subplots(n_show + 1, 2, figsize=(13, 3.0 * (n_show + 1)))
    for r in range(n_show):
        _show(axes[r, 0], x, elev, c_f[r], mask, f"clay c field, realization {r + 1}", "viridis", "c (kPa)", fig)
        _show(axes[r, 1], x, elev, p_f[r], mask, f"sand phi field, realization {r + 1}", "magma", "phi (deg)", fig)
    maps = property_maps(S, {"clay": (c_key, c_f[0]), "sand": (p_key, p_f[0])})
    _show(axes[-1, 0], x, elev, maps["c_kPa"], mask, "c in one CMC layout (realization 1 fields)",
          "viridis", "c (kPa)", fig)
    _show(axes[-1, 1], x, elev, maps["phi_deg"], mask, "phi in the same CMC layout", "magma", "phi (deg)", fig)
    fig.suptitle(rf.describe(), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path.name}")
    return S


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acf", default="markov", choices=ACFS, help="correlation function")
    ap.add_argument("--delta-h", type=float, default=20.0, help="horizontal scale of fluctuation (m)")
    ap.add_argument("--delta-v", type=float, default=2.0, help="vertical scale of fluctuation (m)")
    ap.add_argument("--method", default="cmd", choices=["cmd", "kl"], help="simulation method")
    ap.add_argument("--energy", type=float, default=0.95, help="KL: share of the variance kept")
    ap.add_argument("--no-rescale", action="store_true", help="KL: keep the truncated variance")
    ap.add_argument("--n-show", type=int, default=3, help="realizations shown in the figure")
    ap.add_argument("--verify", action="store_true", help="check correlation and moments")
    ap.add_argument("--n-check", type=int, default=1000, help="realizations used by --verify")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _, _, xs = cs.load_boreholes()
    x, elev, _ = cs.grid(xs)
    rf = SeparableRandomField(x, elev, args.acf, args.delta_h, args.delta_v, args.method,
                              args.energy, rescale=not args.no_rescale)
    print(rf.describe())
    for soil, key in RANDOM_PARAMS.items():
        v = sp.SOIL_PROPERTIES[soil][key]
        mu, s = sp.lognormal_params(v["mean"], v["cov"])
        print(f"  {soil} {key}: lognormal, mean {v['mean']:g}, COV {v['cov']:g} "
              f"(ln mean {mu:.4f}, ln sd {s:.4f})")
    plot_acfs(args.delta_h, args.delta_v, HERE / "random_field_acf.png")
    S = plot_realizations(rf, args.n_show, rng, HERE / "random_field_realizations.png")
    shares = {sp.STATE_NAMES[k]: np.mean(S[S > 0] == k) for k in sp.STATE_NAMES}
    print("  CMC layout used in the figure: " + ", ".join(f"{k} {v:.0%}" for k, v in shares.items()))
    if args.verify:
        verify(rf, args.n_check, rng, HERE / "random_field_check.png")


if __name__ == "__main__":
    main()
