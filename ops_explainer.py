"""
ops_explainer.py
----------------
Rule-based AI narrative generator for hub operational status.

Produces human-readable explanations that an operations manager can read
at a glance — similar in style to a daily ops briefing.

No external APIs or LLMs are required. All logic is deterministic and
based on known logistics thresholds.

Main function:
    generate_ops_explanation(hub_row, bottleneck_rows, health_score) -> str
"""

import pandas as pd


# ── Thresholds ─────────────────────────────────────────────────────────────
RIDER_OVERLOAD_THRESHOLD   = 3.0   # orders per rider
RIDER_CRITICAL_THRESHOLD   = 6.0
DELAY_HIGH_THRESHOLD       = 25.0  # SLA breach %
DELAY_CRITICAL_THRESHOLD   = 40.0
TAT_HIGH_THRESHOLD         = 50.0  # minutes
TAT_CRITICAL_THRESHOLD     = 70.0
DISTANCE_HIGH_THRESHOLD    = 10.0  # km


def _stress_summary(hub_name: str, health_score: float) -> str:
    """One-sentence headline about the hub's overall condition."""
    if health_score >= 70:
        return (
            f"**{hub_name}** is operating within normal parameters. "
            "No immediate intervention required."
        )
    elif health_score >= 40:
        return (
            f"**{hub_name}** is showing signs of operational stress "
            f"(Health Score: {health_score:.0f}/100). "
            "Monitoring and light intervention recommended."
        )
    else:
        return (
            f"⚠️ **{hub_name}** is in a **critical operational state** "
            f"(Health Score: {health_score:.0f}/100). "
            "Immediate corrective action is required."
        )


def _primary_drivers(hub_row: pd.Series) -> list[str]:
    """
    Identify which KPIs are outside acceptable ranges and return
    human-readable descriptions of each driver.
    """
    drivers = []

    orders_incoming  = hub_row["orders_incoming"]
    active_riders    = max(hub_row["active_riders"], 1)
    opr              = orders_incoming / active_riders
    delay_rate       = hub_row["delay_rate_percent"]
    avg_time         = hub_row["avg_delivery_time_minutes"]
    avg_distance     = hub_row["avg_distance_km"]

    # ── Rider load ──
    if opr >= RIDER_CRITICAL_THRESHOLD:
        drivers.append(
            f"🔴 **Severe rider shortage** — {opr:.1f} shipments per delivery partner "
            f"({orders_incoming} shipments, only {int(active_riders)} partners active)"
        )
    elif opr >= RIDER_OVERLOAD_THRESHOLD:
        drivers.append(
            f"🟡 **Rider overload** — {opr:.1f} shipments per delivery partner "
            f"(target: ≤ 3.0)"
        )

    # ── SLA breach / delays ──
    if delay_rate >= DELAY_CRITICAL_THRESHOLD:
        drivers.append(
            f"🔴 **Critical SLA breach rate** — {delay_rate:.1f}% of shipments delayed "
            "(threshold: 20%)"
        )
    elif delay_rate >= DELAY_HIGH_THRESHOLD:
        drivers.append(
            f"🟡 **Elevated SLA breach rate** — {delay_rate:.1f}% of shipments delayed"
        )

    # ── Delivery time ──
    if avg_time >= TAT_CRITICAL_THRESHOLD:
        drivers.append(
            f"🔴 **Very high delivery TAT** — {avg_time:.0f} min average "
            "(best practice: < 45 min)"
        )
    elif avg_time >= TAT_HIGH_THRESHOLD:
        drivers.append(
            f"🟡 **Above-target delivery TAT** — {avg_time:.0f} min average"
        )

    # ── Distance ──
    if avg_distance >= DISTANCE_HIGH_THRESHOLD:
        drivers.append(
            f"🟡 **High average route distance** — {avg_distance:.1f} km per delivery "
            "(consider zone re-clustering)"
        )

    return drivers


def _action_plan(drivers: list[str], health_score: float) -> list[str]:
    """
    Generate a prioritised action plan based on identified drivers.
    """
    actions = []

    for d in drivers:
        if "rider shortage" in d.lower() or "rider overload" in d.lower():
            actions.append(
                "**Rider Reallocation** — Move available delivery partners "
                "from low-load hubs to this hub."
            )
            if "severe" in d.lower():
                actions.append(
                    "**Activate Gig Pool** — Request additional gig riders "
                    "from the partner network."
                )

        if "sla breach" in d.lower() or "delayed" in d.lower():
            if "critical" in d.lower():
                actions.append(
                    "**Temporary Shipment Cap** — Limit new assignments until "
                    "backlog is cleared."
                )
            actions.append(
                "**Priority Re-Queuing** — Push SLA-breach orders to the front "
                "of the dispatch queue."
            )

        if "tat" in d.lower() or "delivery time" in d.lower():
            actions.append(
                "**Route Batching** — Cluster deliveries by PIN code to reduce "
                "per-rider travel distance."
            )

        if "distance" in d.lower():
            actions.append(
                "**Zone Re-Clustering** — Review hub catchment area boundaries "
                "and redistribute distant orders to a nearer hub."
            )

    if not actions:
        actions.append(
            "**Continue Monitoring** — Maintain current operations and review "
            "metrics in the next cycle."
        )

    # Deduplicate while preserving order
    seen, unique_actions = set(), []
    for a in actions:
        key = a[:40]
        if key not in seen:
            seen.add(key)
            unique_actions.append(a)

    return unique_actions


def generate_ops_explanation(
    hub_row: pd.Series,
    health_score: float,
) -> str:
    """
    Generate a structured, human-readable operational briefing for one hub.

    Parameters
    ----------
    hub_row     : pd.Series  — one row from data_generator hub snapshot
    health_score: float      — pre-computed health score for this hub

    Returns
    -------
    str — Markdown-formatted narrative (safe for st.markdown)
    """
    hub_name = hub_row["hub_name"]

    summary  = _stress_summary(hub_name, health_score)
    drivers  = _primary_drivers(hub_row)
    actions  = _action_plan(drivers, health_score)

    lines = [summary, ""]

    if drivers:
        lines.append("**Primary Drivers:**")
        for d in drivers:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("**Suggested Actions:**")
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a}")

    return "\n".join(lines)


def generate_all_explanations(
    hub_df: pd.DataFrame,
    health_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Convenience wrapper — returns a dict mapping hub_name → explanation string.

    Parameters
    ----------
    hub_df    : full hub snapshot DataFrame
    health_df : output of hub_health.calculate_hub_health_score()

    Returns
    -------
    dict[str, str]
    """
    health_map = dict(zip(health_df["hub_name"], health_df["health_score"]))

    result = {}
    for _, row in hub_df.iterrows():
        score = health_map.get(row["hub_name"], 50.0)
        result[row["hub_name"]] = generate_ops_explanation(row, score)

    return result


# ── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_hub_data
    from hub_health     import calculate_hub_health_score

    hub_df    = generate_hub_data(bottleneck_hubs=["Bangalore North", "Delhi Central"])
    health_df = calculate_hub_health_score(hub_df)
    exps      = generate_all_explanations(hub_df, health_df)

    for hub, exp in exps.items():
        print(f"\n{'='*60}")
        print(f"HUB: {hub}")
        print(exp)
