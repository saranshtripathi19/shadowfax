"""
hub_health.py
-------------
Calculates a composite Hub Health Score for each delivery hub.

The health score is a weighted aggregation of four operational KPIs:
    delay_rate_percent        → weight 0.35
    orders_per_rider          → weight 0.30
    avg_delivery_time_minutes → weight 0.20
    rider_shortage_factor     → weight 0.15

Higher raw score = MORE stressed hub.
We invert it so that 100 = perfectly healthy, 0 = critically stressed.

Health categories:
    0  – 40  → Critical  (🔴)
    40 – 70  → Warning   (🟡)
    70 – 100 → Healthy   (🟢)
"""

import pandas as pd
import numpy as np


# ── Weights (must sum to 1.0) ──────────────────────────────────────────────
WEIGHTS = {
    "delay_rate_percent":        0.35,
    "orders_per_rider":          0.30,
    "avg_delivery_time_minutes": 0.20,
    "rider_shortage_factor":     0.15,
}

# ── Expected "worst-case" ceilings used for per-metric normalisation ───────
# Values at or above the ceiling map to a raw score of 1.0 (maximum stress)
CEILINGS = {
    "delay_rate_percent":        60.0,   # 60 % SLA breach = worst case
    "orders_per_rider":          10.0,   # 10 orders/rider = severe overload
    "avg_delivery_time_minutes": 90.0,   # 90 min TAT = very bad
    "rider_shortage_factor":     1.0,    # factor is already in [0, 1]
}


def calculate_hub_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 0-100 health score for every hub in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Hub snapshot from data_generator.generate_hub_data().
        Required columns:
            hub_name, orders_incoming, active_riders,
            delay_rate_percent, avg_delivery_time_minutes

    Returns
    -------
    pd.DataFrame
        Columns: hub_name, health_score, health_status
            health_score  – float in [0, 100]; higher = healthier
            health_status – "Healthy" | "Warning" | "Critical"
    """
    work = df.copy()

    # ── Derive helper columns ──────────────────────────────────────────────
    # orders_per_rider: main congestion metric
    work["orders_per_rider"] = (
        work["orders_incoming"] / work["active_riders"].clip(lower=1)
    )

    # rider_shortage_factor: fraction of orders that CANNOT be served by
    # current riders assuming each rider handles max 4 deliveries/shift.
    # 0 = no shortage, 1 = total shortage
    max_throughput = work["active_riders"] * 4
    work["rider_shortage_factor"] = (
        (work["orders_incoming"] - max_throughput)
        .clip(lower=0)
        / work["orders_incoming"].clip(lower=1)
    ).clip(upper=1.0)

    # ── Per-metric normalised stress score [0, 1] ──────────────────────────
    stress_components = {}
    for metric, ceiling in CEILINGS.items():
        stress_components[metric] = (work[metric] / ceiling).clip(upper=1.0)

    # ── Weighted sum of stress (0 = perfectly healthy, 1 = critical) ──────
    raw_stress = sum(
        WEIGHTS[m] * stress_components[m] for m in WEIGHTS
    )

    # ── Invert: health = 100 − (stress × 100), rounded to 1 dp ───────────
    work["health_score"] = (1.0 - raw_stress) * 100
    work["health_score"] = work["health_score"].clip(0, 100).round(1)

    # ── Categorise ────────────────────────────────────────────────────────
    def _status(score: float) -> str:
        if score >= 70:
            return "Healthy"
        elif score >= 40:
            return "Warning"
        else:
            return "Critical"

    work["health_status"] = work["health_score"].apply(_status)

    result = work[["hub_name", "health_score", "health_status"]].copy()
    # Sort worst → best so dashboards can show most critical hubs first
    result = result.sort_values("health_score").reset_index(drop=True)
    return result


# ── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_hub_data
    df = generate_hub_data(bottleneck_hubs=["Bangalore North", "Delhi Central"])
    health = calculate_hub_health_score(df)
    print(health.to_string(index=False))
