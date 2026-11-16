"""Print the full problem set-up (geometry, soils, site data, CMC inputs,
Monte Carlo settings). The figure of the slope and boreholes is made by
plot_slope_profile.py.

Usage:  python show_problem.py
Requires numpy, pandas.
"""
import numpy as np

import cmc_setup as cs
import slope_geometry as geo
import soil_properties as sp


def main():
    np.set_printoptions(precision=4, suppress=True)
    df, logs, xs = cs.load_boreholes()
    surf = cs.load_surface()
    x, elev, cols = cs.grid(xs)
    Pv, Ph = cs.transition_matrices()

    print("=== Slope geometry ===")
    print(f"crest ({geo.CREST_X}, {geo.CREST_ELEV}) m, toe ({geo.TOE_X}, {geo.TOE_ELEV}) m, "
          f"H = {geo.SLOPE_HEIGHT} m, angle = {geo.SLOPE_ANGLE_DEG:.1f} deg, "
          f"base at {geo.BASE_ELEV} m, x in [{geo.X_MIN}, {geo.X_MAX}] m")
    print("\n=== Soil properties ===")
    for code in sp.STATE_NAMES:
        print(f"{code} {sp.describe(code)}")
    print("\n=== Site data ===")
    for b, g in df.groupby("borehole_id"):
        print(f"{b}: x = {xs[b]:.0f} m, collar {g['collar_elev_m'].iloc[0]} m, "
              f"{g['state'].notna().sum()} logged samples")
    print(f"ground surface soil: {len(surf)} points "
          f"({', '.join(f'{s}: {n}' for s, n in surf['segment'].value_counts().items())})")
    print("\n=== CMC ===")
    print(f"grid {len(elev)} x {len(x)} (dz {cs.DZ} m, dx {cs.DX} m), borehole columns {cols}")
    print("VTPM (dz = 0.3 m):\n", Pv)
    print("HTPM (dx = 2 m):\n", Ph)
    print(f"\nMonte Carlo: N = {cs.N_SIM_MOMENTS} for mean/std of FS, N = {cs.N_SIM_PF} for Pf = P(FS < {cs.FS_FAILURE})")


if __name__ == "__main__":
    main()
