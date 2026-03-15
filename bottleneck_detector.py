"""
bottleneck_detector.py
----------------------
Analyzes hub-level logistics data and flags operational bottlenecks.

Three bottleneck types are detected:
  1. Rider Shortage       – orders_per_rider > 3
  2. High Delivery Delays – delay_rate_percent > 20
  3. Routing Inefficiency – avg_delivery_time_minutes > 45

Each detected issue is assigned a severity:
  Low    → mild threshold breach
  Medium → moderate breach
  High   → severe breach (multiple thresholds or extreme values)

The output dataframe is consumed by the recommendation engine and the dashboard.
"""

import pandas as pd


# ──────────────────────────────────────────────
# Threshold constants (easy to tune)
# ──────────────────────────────────────────────
RIDER_SHORTAGE_THRESHOLD   = 3.0   # orders per rider
DELAY_RATE_THRESHOLD       = 20.0  # percent
DELIVERY_TIME_THRESHOLD    = 45.0  # minutes


def _severity(value: float, low: float, medium: float) -> str:
    """
    Map a numeric value to a severity label using two breakpoints.

    Parameters
    ----------
    value  : The metric value to classify.
    low    : Threshold above which severity is 'Low'.
    medium : Threshold above which severity is 'Medium' (else 'High').

    Returns
    -------
    'Low' | 'Medium' | 'High'
    """
    if value <= low:
        return "Low"
    elif value <= medium:
        return "Medium"
    else:
        return "High"


def detect_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify hubs with operational bottlenecks and classify their severity.

    Parameters
    ----------
    df : pd.DataFrame
        Hub snapshot produced by data_generator.generate_hub_data().
        Required columns:
          hub_name, orders_incoming, active_riders,
          delay_rate_percent, avg_delivery_time_minutes

    Returns
    -------
    pd.DataFrame
        One row per detected issue with columns:
          hub_name, issue_type, severity, orders_per_rider, delay_rate
        Returns an empty dataframe if no bottlenecks are found.
    """
    issues = []

    for _, row in df.iterrows():
        hub = row["hub_name"]

        # Guard against division by zero
        riders = max(row["active_riders"], 1)

        orders_per_rider  = row["orders_incoming"] / riders
        delay_rate        = row["delay_rate_percent"]
        avg_delivery_time = row["avg_delivery_time_minutes"]

        # ── Rule 1: Rider Shortage ──
        if orders_per_rider > RIDER_SHORTAGE_THRESHOLD:
            sev = _severity(orders_per_rider, low=4.0, medium=6.0)
            issues.append({
                "hub_name":        hub,
                "issue_type":      "Rider Shortage",
                "severity":        sev,
                "orders_per_rider": round(orders_per_rider, 2),
                "delay_rate":       round(delay_rate, 1),
            })

        # ── Rule 2: High Delivery Delays ──
        if delay_rate > DELAY_RATE_THRESHOLD:
            sev = _severity(delay_rate, low=25.0, medium=35.0)
            issues.append({
                "hub_name":        hub,
                "issue_type":      "High Delivery Delays",
                "severity":        sev,
                "orders_per_rider": round(orders_per_rider, 2),
                "delay_rate":       round(delay_rate, 1),
            })

        # ── Rule 3: Routing Inefficiency ──
        if avg_delivery_time > DELIVERY_TIME_THRESHOLD:
            sev = _severity(avg_delivery_time, low=55.0, medium=70.0)
            issues.append({
                "hub_name":        hub,
                "issue_type":      "Routing Inefficiency",
                "severity":        sev,
                "orders_per_rider": round(orders_per_rider, 2),
                "delay_rate":       round(delay_rate, 1),
            })

    if not issues:
        # Return empty but schema-consistent dataframe
        return pd.DataFrame(columns=[
            "hub_name", "issue_type", "severity", "orders_per_rider", "delay_rate"
        ])

    result = pd.DataFrame(issues)

    # Sort: High → Medium → Low for dashboard priority display
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    result["_sort"] = result["severity"].map(severity_order)
    result = result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    return result


# ──────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_hub_data
    df = generate_hub_data(bottleneck_hubs=["Bangalore North", "Delhi Central"])
    bottlenecks = detect_bottlenecks(df)
    print(bottlenecks.to_string(index=False))
