"""
data_generator.py
-----------------
Generates synthetic logistics data for 10 last-mile delivery hubs.

Each row of the dataframe represents a hub's current operational snapshot.
The generator can produce:
  - Normal operating conditions
  - Bottleneck scenarios (high orders, low riders, or spiked delivery times)

This module is the foundation for all downstream analysis components.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────
# Hub definitions (name + city mapping)
# ──────────────────────────────────────────────
HUBS = [
    {"hub_name": "Bangalore North",   "city": "Bangalore"},
    {"hub_name": "Bangalore East",    "city": "Bangalore"},
    {"hub_name": "Bangalore South",   "city": "Bangalore"},
    {"hub_name": "Bangalore West",    "city": "Bangalore"},
    {"hub_name": "Delhi Central",     "city": "Delhi"},
    {"hub_name": "Delhi South",       "city": "Delhi"},
    {"hub_name": "Delhi East",        "city": "Delhi"},
    {"hub_name": "Mumbai North",      "city": "Mumbai"},
    {"hub_name": "Mumbai South",      "city": "Mumbai"},
    {"hub_name": "Hyderabad Central", "city": "Hyderabad"},
]


def generate_hub_data(
    bottleneck_hubs: list[str] | None = None,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Generate a synthetic snapshot of hub-level logistics data.

    Parameters
    ----------
    bottleneck_hubs : list[str] | None
        List of hub names where an artificial bottleneck should be injected.
        Bottleneck types are randomly chosen from:
          - "high_orders"   → surge in incoming orders
          - "low_riders"    → rider shortage
          - "delay_spike"   → spike in delivery times and delay rate
        Pass None (default) to let the generator introduce random, mild
        fluctuations without forced bottlenecks.
    seed : int | None
        Random seed for reproducibility. Pass None for fully random output.

    Returns
    -------
    pd.DataFrame
        One row per hub with the following columns:
          hub_name, city, orders_incoming, orders_processed,
          active_riders, avg_delivery_time_minutes,
          delay_rate_percent, avg_distance_km, timestamp
    """
    rng = np.random.default_rng(seed)  # reproducible randomness when seed given

    records = []

    for hub in HUBS:
        name = hub["hub_name"]
        city = hub["city"]

        # ── Base values (realistic ranges for a tier-1 Indian logistics hub) ──
        orders_incoming           = int(rng.integers(80, 200))
        active_riders             = int(rng.integers(30, 70))
        avg_delivery_time_minutes = float(rng.uniform(25, 45))
        delay_rate_percent        = float(rng.uniform(5, 18))
        avg_distance_km           = float(rng.uniform(3, 12))

        # ── Inject bottleneck if this hub is flagged ──
        if bottleneck_hubs and name in bottleneck_hubs:
            scenario = rng.choice(["high_orders", "low_riders", "delay_spike"])

            if scenario == "high_orders":
                # Surge: orders jump 2-3× while riders stay flat
                orders_incoming = int(orders_incoming * rng.uniform(2.2, 3.0))

            elif scenario == "low_riders":
                # Rider shortage: available riders drop to ≤ 30 % of normal
                active_riders = max(5, int(active_riders * rng.uniform(0.15, 0.30)))

            else:  # delay_spike
                # Routing / traffic issues: delivery time and delay rate spike
                avg_delivery_time_minutes = float(rng.uniform(55, 90))
                delay_rate_percent        = float(rng.uniform(28, 55))

        # ── Derive orders_processed (always ≤ orders_incoming) ──
        # Throughput is limited by rider capacity (approx 3-4 deliveries/hour)
        max_capacity    = active_riders * 4
        orders_processed = int(min(orders_incoming, rng.integers(
            int(max_capacity * 0.7), max(int(max_capacity * 0.7) + 1, max_capacity + 1)
        )))

        records.append({
            "hub_name":                  name,
            "city":                       city,
            "orders_incoming":            orders_incoming,
            "orders_processed":           orders_processed,
            "active_riders":              active_riders,
            "avg_delivery_time_minutes":  round(avg_delivery_time_minutes, 1),
            "delay_rate_percent":         round(delay_rate_percent, 1),
            "avg_distance_km":            round(avg_distance_km, 2),
            "timestamp":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    df = pd.DataFrame(records)
    return df


def generate_timeseries_data(
    hours: int = 24,
    bottleneck_hubs: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate hourly logistics data for the last *hours* hours across all hubs.

    This produces a long-format DataFrame suitable for time-series line charts.
    Each hub gets one row per hour, with smooth correlated noise simulating
    real intraday patterns (morning/evening order spikes, afternoon lull).

    Parameters
    ----------
    hours           : number of hourly data points per hub (default 24)
    bottleneck_hubs : optional list of hubs to stress at peak hours
    seed            : random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Columns: hub_name, city, timestamp, orders_incoming, active_riders,
                 delay_rate_percent, avg_delivery_time_minutes
        Shape  : len(HUBS) × hours rows
    """
    from datetime import timedelta

    rng         = np.random.default_rng(seed)
    now         = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps  = [now - timedelta(hours=h) for h in range(hours - 1, -1, -1)]

    # Intraday demand multiplier: peaks at 9-11 AM and 6-8 PM
    def _demand_mult(hour_of_day: int) -> float:
        if 9 <= hour_of_day <= 11:
            return 1.5
        elif 18 <= hour_of_day <= 20:
            return 1.3
        elif 2 <= hour_of_day <= 5:
            return 0.4   # early-morning quiet period
        else:
            return 1.0

    records = []

    for hub in HUBS:
        name = hub["hub_name"]
        city = hub["city"]

        # Base values for this hub (slightly randomised per hub)
        base_orders  = float(rng.integers(100, 180))
        base_riders  = float(rng.integers(35, 65))
        base_delay   = float(rng.uniform(8,  18))
        base_tat     = float(rng.uniform(28, 42))

        for ts in timestamps:
            hour_of_day = ts.hour
            mult = _demand_mult(hour_of_day)

            # Smooth noise via small random walk component
            noise_o = float(rng.normal(0, 12))
            noise_r = float(rng.normal(0, 3))
            noise_d = float(rng.normal(0, 3))
            noise_t = float(rng.normal(0, 4))

            orders  = max(10, int(base_orders * mult + noise_o))
            riders  = max(5,  int(base_riders + noise_r))
            delay   = float(np.clip(base_delay * mult + noise_d, 2, 90))
            tat     = float(np.clip(base_tat   * mult + noise_t, 15, 120))

            # Inject stress at peak hours for bottleneck hubs
            if bottleneck_hubs and name in bottleneck_hubs and mult > 1.0:
                orders  = int(orders  * rng.uniform(1.5, 2.2))
                riders  = max(5, int(riders * rng.uniform(0.4, 0.6)))
                delay   = float(np.clip(delay  * rng.uniform(1.5, 2.0), 0, 90))
                tat     = float(np.clip(tat    * rng.uniform(1.3, 1.8), 0, 120))

            records.append({
                "hub_name":                   name,
                "city":                        city,
                "timestamp":                   ts,
                "orders_incoming":             orders,
                "active_riders":               riders,
                "delay_rate_percent":          round(delay, 1),
                "avg_delivery_time_minutes":   round(tat,   1),
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_hub_data(bottleneck_hubs=["Bangalore North", "Delhi Central"])
    print("Snapshot:")
    print(df.to_string(index=False))

    ts = generate_timeseries_data(hours=6, bottleneck_hubs=["Bangalore North"])
    print("\nTimeseries (first 12 rows):")
    print(ts.head(12).to_string(index=False))
