"""Soil properties for the slope-reliability problem.

Two sources of uncertainty:
  1. geological uncertainty - WHICH soil type occupies each cell, simulated
     by the coupled Markov chain;
  2. parameter uncertainty - the clay cohesion and the sand friction angle
     are random variables or random field; the two are
     independent of each other and of the soil layout.

State codes are the same as in boreholes_slope.csv: 1 = clay, 2 = sand, 3 = silt.

    soil   gamma (kN/m3)   c (kPa)                              phi (deg)
    clay   20              lognormal, mean 40, COV 0.36          0 (fixed)
    sand   20              2 (fixed)                            lognormal, mean 33, COV 0.18
    silt   20              6 (fixed)                            28 (fixed)

Shear strength: Mohr-Coulomb, tau_f = c + sigma'_n * tan(phi).
Failure: FS < 1.
"""
import numpy as np

STATE_NAMES = {1: "clay", 2: "sand", 3: "silt"}

# Each parameter is either a fixed number or a dict describing a random variable.
SOIL_PROPERTIES = {
    "clay": {"gamma_kN_m3": 20.0,
             "c_kPa": {"dist": "lognormal", "mean": 40.0, "cov": 0.36},
             "phi_deg": 0.0},
    "sand": {"gamma_kN_m3": 20.0,
             "c_kPa": 2.0,
             "phi_deg": {"dist": "lognormal", "mean": 33.0, "cov": 0.18}},
    "silt": {"gamma_kN_m3": 20.0,
             "c_kPa": 6.0,
             "phi_deg": 28.0},
}


def is_random(value):
    return isinstance(value, dict)


def mean_value(value):
    return value["mean"] if is_random(value) else float(value)


def lognormal_params(mean, cov):
    """(mu, sigma) of ln X for a lognormal X with the given mean and COV."""
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    return np.log(mean) - 0.5 * sigma ** 2, sigma


def sample(value, n, rng):
    """n samples of a parameter (fixed values are repeated)."""
    if not is_random(value):
        return np.full(n, float(value))
    m, sd = value["mean"], value["mean"] * value["cov"]
    if value["dist"] == "normal":
        return rng.normal(m, sd, n)
    if value["dist"] == "lognormal":
        mu, s = lognormal_params(m, value["cov"])
        return rng.lognormal(mu, s, n)
    raise ValueError(value["dist"])


def property_table():
    """{state_code: {gamma_kN_m3, c_kPa, phi_deg (mean values), name}}."""
    return {code: {k: mean_value(v) for k, v in SOIL_PROPERTIES[name].items()} | {"name": name}
            for code, name in STATE_NAMES.items()}


def sample_properties(n, rng):
    """n realizations of the soil parameters.
    Returns {state_code: {"gamma_kN_m3": (n,), "c_kPa": (n,), "phi_deg": (n,)}}."""
    return {code: {k: sample(v, n, rng) for k, v in SOIL_PROPERTIES[name].items()}
            for code, name in STATE_NAMES.items()}


def describe(code):
    """One-line text description of a soil's parameters."""
    name = STATE_NAMES[code]
    parts = []
    for key, label, unit in (("gamma_kN_m3", "gamma", "kN/m3"), ("c_kPa", "c", "kPa"), ("phi_deg", "phi", "deg")):
        v = SOIL_PROPERTIES[name][key]
        if is_random(v):
            parts.append(f"{label}: {v['dist']}, mean {v['mean']:g} {unit}, COV {v['cov']:g}")
        else:
            parts.append(f"{label} = {v:g} {unit}")
    return f"{name}: " + ", ".join(parts)


def property_arrays(states, key):
    """Map an integer array of state codes (1/2/3) to MEAN property values.
    Cells with a code outside 1..3 (e.g. 0 = outside the slope) give NaN."""
    table = property_table()
    states = np.asarray(states)
    out = np.full(states.shape, np.nan)
    for code, p in table.items():
        out[states == code] = p[key]
    return out


if __name__ == "__main__":
    for code in STATE_NAMES:
        print(code, describe(code))
    s = sample_properties(100_000, np.random.default_rng(0))
    print(f"check: clay c mean {s[1]['c_kPa'].mean():.2f}, COV {s[1]['c_kPa'].std() / s[1]['c_kPa'].mean():.3f}; "
          f"sand phi mean {s[2]['phi_deg'].mean():.2f}, COV {s[2]['phi_deg'].std() / s[2]['phi_deg'].mean():.3f}")
