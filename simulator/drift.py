"""Drift injection for the traffic simulator, plus a simple check that drift is visible.

inject_drift() makes the incoming bookings look like a different population:
  - lead_time:      booked much further ahead (x2.5, plus noise)
  - adr:            prices 40% higher
  - market_segment: 60% of bookings switched to "Groups"
  - country:        half of the Portuguese bookings become foreign (GBR, DEU, FRA)
These are features the model relies on, so its predictions should move too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FOREIGN = ["GBR", "DEU", "FRA"]


def inject_drift(df: pd.DataFrame, seed: int = 0, strength: float = 1.0) -> pd.DataFrame:
    """Return a drifted copy of df. strength=0 returns an unchanged copy, 1 is the full shift."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if strength <= 0 or n == 0:
        return out

    lead = out["lead_time"].astype(float)
    out["lead_time"] = np.clip(
        (lead * (1 + 1.5 * strength) + rng.normal(0, 10, n)).round(), 0, 800
    ).astype(int)
    out["adr"] = np.clip(out["adr"].astype(float) * (1 + 0.4 * strength), 0, 1000).round(2)

    to_groups = rng.random(n) < 0.6 * strength
    out.loc[to_groups, "market_segment"] = "Groups"

    is_prt = (out["country"] == "PRT").to_numpy()
    to_foreign = is_prt & (rng.random(n) < 0.5 * strength)
    out.loc[to_foreign, "country"] = rng.choice(FOREIGN, to_foreign.sum())
    return out


def numeric_drift(reference: pd.Series, current: pd.Series, alpha: float = 0.01) -> bool:
    """Two-sample Kolmogorov-Smirnov test: True when the distributions differ."""
    return stats.ks_2samp(reference.astype(float), current.astype(float)).pvalue < alpha


def categorical_drift(reference: pd.Series, current: pd.Series, threshold: float = 0.1) -> bool:
    """Total variation distance between category shares: True when it exceeds the threshold."""
    p = reference.value_counts(normalize=True)
    q = current.value_counts(normalize=True)
    tvd = 0.5 * p.subtract(q, fill_value=0).abs().sum()
    return tvd > threshold
