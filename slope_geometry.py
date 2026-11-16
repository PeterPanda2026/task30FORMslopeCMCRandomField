"""Slope geometry for the slope-reliability problem with geological uncertainty.

Coordinates are those of boreholes_slope.csv:
    x          : horizontal distance along the section (m), model width [0, 70] m
    elevation  : m; the crest ground surface is at 28.2 m

Slope:
    slope height        : H = 13.5 m
    slope length (horiz): L = 33.75 m  -> 1V:2.5H slope, angle = atan(0.4) = 21.8 deg
    crest               : (15.0, 28.2) m
    toe                 : (48.75, 14.7) m
    ground surface      : 28.2 m left of the crest, 14.7 m right of the toe
    model base          : elevation 1.2 m -> model height 27.0 m
    z                   : elevation - 1.2 m (height above the model base)

Boreholes: BH1 (x = 0 m) is drilled from the crest level (collar 28.2 m),
BH2 (x = 20 m) and BH3 (x = 40 m) from the slope face (collars 26.2 m and
18.2 m), BH4 (x = 50 m) and BH5 (x = 70 m) from the toe level (collar 14.7 m).
The soil type along the whole ground surface (crest, face, toe) is given in
ground_surface_soil.csv. Only CMC cells whose centre is inside the slope
polygon are soil.

Dry slope (no pore pressure).
"""
import numpy as np

SLOPE_HEIGHT = 13.5            # m
SLOPE_GRADIENT = 0.4           # tan(slope angle) = H / L
SLOPE_LENGTH = SLOPE_HEIGHT / SLOPE_GRADIENT   # 33.75 m, horizontal projection of the face
SLOPE_ANGLE_DEG = np.degrees(np.arctan(SLOPE_GRADIENT))   # 21.8 deg

X_MIN, X_MAX = 0.0, 70.0                 # m, BH1 to BH5
CREST_ELEV = 28.2                        # m
TOE_ELEV = CREST_ELEV - SLOPE_HEIGHT     # 14.7 m
BASE_ELEV = 1.2                          # m
MODEL_HEIGHT = CREST_ELEV - BASE_ELEV    # 27.0 m
CREST_X = 15.0                           # m
TOE_X = CREST_X + SLOPE_LENGTH           # 48.75 m

BOREHOLE_X = {"BH1": 0.0, "BH2": 20.0, "BH3": 40.0, "BH4": 50.0, "BH5": 70.0}

# Slope polygon, counter-clockwise (x, elevation)
SLOPE_POLYGON = np.array([
    [X_MIN, BASE_ELEV],
    [X_MAX, BASE_ELEV],
    [X_MAX, TOE_ELEV],
    [TOE_X, TOE_ELEV],
    [CREST_X, CREST_ELEV],
    [X_MIN, CREST_ELEV],
])

WATER_TABLE = None   # dry slope (no pore pressure)


def ground_surface(x):
    """Ground-surface elevation (m) of the slope at horizontal position x (m)."""
    x = np.asarray(x, dtype=float)
    face = CREST_ELEV - (x - CREST_X) * SLOPE_HEIGHT / SLOPE_LENGTH
    return np.where(x <= CREST_X, CREST_ELEV, np.where(x >= TOE_X, TOE_ELEV, face))


def inside_slope(x, elev):
    """True where the point (x, elev) lies inside the slope body."""
    x, elev = np.asarray(x, float), np.asarray(elev, float)
    return (x >= X_MIN) & (x <= X_MAX) & (elev >= BASE_ELEV) & (elev <= ground_surface(x))


def to_model_z(elev):
    """Convert elevation (m) to z (m above the model base)."""
    return np.asarray(elev, float) - BASE_ELEV


def to_elevation(z):
    """Convert z (m above the model base) to elevation (m)."""
    return np.asarray(z, float) + BASE_ELEV


if __name__ == "__main__":
    print(f"Slope: H = {SLOPE_HEIGHT} m, L = {SLOPE_LENGTH} m, angle = {SLOPE_ANGLE_DEG:.1f} deg")
    print(f"Crest (x, elev) = ({CREST_X}, {CREST_ELEV}) m, toe = ({TOE_X}, {TOE_ELEV}) m")
    print(f"Model: x in [{X_MIN}, {X_MAX}] m, elevation in [{BASE_ELEV}, {CREST_ELEV}] m")
    print("Slope polygon (x, elev):")
    for p in SLOPE_POLYGON:
        print(f"  ({p[0]:5.1f}, {p[1]:5.1f})")
    for b, x in BOREHOLE_X.items():
        print(f"  {b}: x = {x:4.1f} m, ground surface = {float(ground_surface(x)):.1f} m")
