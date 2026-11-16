"""Plot the slope profile and the borehole contents.

Usage:  python plot_slope_profile.py [--no-show]
Saves slope_profile.png.

Each 0.3 m borehole sample at elevation e is drawn as the interval from e
down to e - 0.3 m; the top interval is extended up to the collar, so every
column starts at the ground surface. Numbers to the right of each column are
the elevations (m) of its collar, its layer boundaries and its bottom (the
model base, 1.2 m). Coloured squares on the ground surface show the mapped
surface soil (ground_surface_soil.csv). The legend gives the unit weight,
cohesion and friction angle of each soil (random variables with their
distribution, mean and COV).
Requires numpy, pandas, matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

import slope_geometry as geo
import soil_properties as sp

HERE = Path(__file__).resolve().parent
COLORS = {1: "#e8908a", 2: "#f2d27a", 3: "#9fb4ee"}     # clay, sand, silt
COL_W = 1.6          # m, drawn width of a borehole column
DZ = 0.3
MIN_LABEL_GAP = 0.8  # m, minimum vertical spacing of boundary labels


def layers(g):
    """Return list of (top, bottom, state) intervals of one borehole, clipped at the model base."""
    g = g.dropna(subset=["state"]).sort_values("elevation_m", ascending=False)
    out = []
    for e, s in zip(g["elevation_m"], g["state"].astype(int)):
        top, bot = e, max(e - DZ, geo.BASE_ELEV)
        if top <= geo.BASE_ELEV + 1e-9:
            break
        if out and out[-1][2] == s and abs(out[-1][1] - top) < 1e-6:
            out[-1][1] = bot
        else:
            out.append([top, bot, s])
    if out:   # the column starts at the collar (ground surface), even if the first sample is lower
        out[0][0] = max(out[0][0], float(g["collar_elev_m"].iloc[0]))
    return out


def spread_labels(values, gap):
    """Shift label positions (sorted high -> low) so neighbours are >= gap apart."""
    pos = list(values)
    for k in range(1, len(pos)):
        pos[k] = min(pos[k], pos[k - 1] - gap)
    # if pushed below the lowest true value, shift the whole stack up
    shift = values[-1] - pos[-1]
    if shift > 0:
        pos = [p + shift for p in pos]
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = max(pos[k], pos[k + 1] + gap)
    return pos


def param_text(label, value, unit):
    if sp.is_random(value):
        return (f"{label}: {value['dist'].capitalize()}, $\\mu$ = {value['mean']:g} {unit}, "
                f"COV = {value['cov']:g}")
    return f"{label} = {value:g} {unit} (fixed)"


def legend_text(code):
    name = sp.STATE_NAMES[code]
    p = sp.SOIL_PROPERTIES[name]
    c_line = param_text("c", p["c_kPa"], "kPa")
    phi_line = param_text("$\\phi$", p["phi_deg"], "deg")
    return (f"{name.capitalize()}, $\\gamma$ = {p['gamma_kN_m3']:g} kN/m$^3$\n"
            f"  {c_line}\n"
            f"  {phi_line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    bh = pd.read_csv(HERE / "boreholes_slope.csv")
    surf = pd.read_csv(HERE / "ground_surface_soil.csv")

    fig, ax = plt.subplots(figsize=(15, 6.8))
    ax.add_patch(Polygon(geo.SLOPE_POLYGON, closed=True, facecolor="#f1f1f1", edgecolor="none"))
    ax.plot([geo.X_MIN, geo.X_MAX], [geo.BASE_ELEV] * 2, "k-", lw=1.2)
    gs_x = [geo.X_MIN, geo.CREST_X, geo.TOE_X, geo.X_MAX]
    gs_z = [geo.CREST_ELEV, geo.CREST_ELEV, geo.TOE_ELEV, geo.TOE_ELEV]
    ax.plot(gs_x, gs_z, "k-", lw=2.5, zorder=4)
    ax.plot([geo.X_MIN] * 2, [geo.BASE_ELEV, geo.CREST_ELEV], color="0.4", lw=0.8, ls="--")
    ax.plot([geo.X_MAX] * 2, [geo.BASE_ELEV, geo.TOE_ELEV], color="0.4", lw=0.8, ls="--")

    # mapped surface soil
    ax.scatter(surf["x_m"], surf["elevation_m"], marker="s", s=16,
               c=[COLORS[s] for s in surf["state"]], edgecolors="k", linewidths=0.4, zorder=5)

    # ground-surface label (corner points of the ground line)
    corners = " - ".join(f"({x:g}, {z:g})" for x, z in zip(gs_x, gs_z))
    ax.text(38, 25.5, "ground surface\n" + corners, ha="center", va="bottom", fontsize=8.5)

    # boreholes
    for b, g in bh.groupby("borehole_id"):
        xb = float(g["x_m"].iloc[0])
        lay = layers(g)
        for top, bot, s in lay:
            ax.add_patch(Rectangle((xb - COL_W / 2, bot), COL_W, top - bot,
                                   facecolor=COLORS[s], edgecolor="k", lw=0.7, zorder=3))
        bounds = [lay[0][0]] + [l[1] for l in lay]
        ypos = spread_labels(bounds, MIN_LABEL_GAP)
        for v, y in zip(bounds, ypos):
            ax.annotate(f"{v:g}", (xb + COL_W / 2, v), (xb + COL_W / 2 + 0.5, y),
                        fontsize=7, va="center", ha="left",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"),
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3") if abs(v - y) > 0.05 else None,
                        zorder=6)
        top = g["collar_elev_m"].iloc[0]
        if geo.CREST_X < xb < geo.TOE_X:      # drilled from the slope face: label clear of the face
            ax.text(xb + 1.2, top + 1.6, f"{b}\nx = {xb:g}", ha="left", va="bottom", fontsize=9,
                    fontweight="bold")
        else:
            ax.text(xb, top + 1.0, f"{b}\nx = {xb:g}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    # unknown zones between boreholes
    for xu, zu in [(10, 14), (30, 14), (45, 7), (60, 8)]:
        ax.text(xu, zu, "soil profile\nunknown", ha="center", va="center",
                fontsize=10, style="italic", color="0.45")

    handles = [Patch(facecolor=COLORS[k], edgecolor="k", label=legend_text(k)) for k in (1, 2, 3)]
    handles.append(Line2D([], [], marker="s", ls="none", markerfacecolor="w", markeredgecolor="k",
                          markersize=5, label="mapped ground-surface soil"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, labelspacing=1.1,
              handlelength=2.2, borderpad=0.8, framealpha=1.0)

    ax.set_title("Slope with five boreholes (state 1 = clay, 2 = sand, 3 = silt; "
                 "elevations of layer boundaries in m)")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_xlim(-4, 74)
    ax.set_ylim(-0.5, 33)
    ax.set_xticks(np.arange(0, 71, 10))
    fig.tight_layout()
    out = HERE / "slope_profile.png"
    fig.savefig(out, dpi=150, metadata={"Software": None})
    print("saved", out)
    if not args.no_show and matplotlib.get_backend().lower() not in {"agg", "pdf", "svg", "ps"}:
        plt.show()


if __name__ == "__main__":
    main()
