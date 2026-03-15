"""
recommendation_engine.py
------------------------
Converts detected bottlenecks into concrete, actionable recommendations
for operations managers.

The engine uses a rule-based approach (if-then logic) to map issue types
to recommended corrective actions.  This is intentional: in logistics ops,
rule-based systems are auditable, explainable, and robust even in edge cases.

Input : bottlenecks dataframe (from bottleneck_detector.detect_bottlenecks)
        full hub dataframe    (from data_generator.generate_hub_data)
Output: recommendations dataframe with columns:
          hub_name | problem | recommended_action | priority
"""

import pandas as pd
import numpy as np



def generate_recommendations(
    bottleneck_df: pd.DataFrame,
    hub_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate operational recommendations for each detected bottleneck.

    Parameters
    ----------
    bottleneck_df : pd.DataFrame
        Output of bottleneck_detector.detect_bottlenecks().
        Required columns: hub_name, issue_type, severity, orders_per_rider, delay_rate

    hub_df : pd.DataFrame
        Full hub snapshot from data_generator.generate_hub_data().
        Used to identify *donor* hubs (low-load hubs that can spare riders).

    Returns
    -------
    pd.DataFrame
        Columns: hub_name, problem, recommended_action, priority
        Returns an empty dataframe if bottleneck_df is empty.
    """
    if bottleneck_df.empty:
        return pd.DataFrame(columns=["hub_name", "problem", "recommended_action", "priority"])

    # ── Pre-compute orders_per_rider for all hubs (used for donor selection) ──
    hub_load = hub_df.copy()
    hub_load["orders_per_rider"] = (
        hub_load["orders_incoming"] / hub_load["active_riders"].clip(lower=1)
    )
    # Donor hub = hub with the smallest load (best candidate to lend riders)
    donor_hub = hub_load.loc[hub_load["orders_per_rider"].idxmin(), "hub_name"]

    recommendations = []

    for _, row in bottleneck_df.iterrows():
        hub       = row["hub_name"]
        issue     = row["issue_type"]
        severity  = row["severity"]
        opr       = row["orders_per_rider"]   # orders per rider

        # ── Map severity → urgency label ──
        priority = {
            "High":   "🔴 Urgent",
            "Medium": "🟡 High",
            "Low":    "🟢 Normal",
        }.get(severity, "Normal")

        # ───────────────────────────────────────────────
        # Rule 1 – Rider Shortage
        # ───────────────────────────────────────────────
        if issue == "Rider Shortage":
            # Estimate riders needed to bring load down to ≤ 3 orders/rider
            hub_orders      = hub_df.loc[hub_df["hub_name"] == hub, "orders_incoming"].values[0]
            current_riders  = hub_df.loc[hub_df["hub_name"] == hub, "active_riders"].values[0]
            target_riders   = max(1, int(hub_orders / 3.0))
            deficit         = max(0, target_riders - current_riders)

            if deficit > 0 and donor_hub != hub:
                action = (
                    f"Reallocate {deficit} rider(s) from **{donor_hub}** to **{hub}**. "
                    f"Target: ≤ 3 orders/rider (currently {opr:.1f})."
                )
            elif deficit == 0:
                action = (
                    f"Load is borderline ({opr:.1f} orders/rider). "
                    "Monitor and prepare standby riders."
                )
            else:
                action = (
                    f"Activate {deficit} gig riders via partner network for **{hub}**."
                )

        # ───────────────────────────────────────────────
        # Rule 2 – High Delivery Delays
        # ───────────────────────────────────────────────
        elif issue == "High Delivery Delays":
            delay = row["delay_rate"]
            if severity == "High":
                action = (
                    f"Impose temporary order cap at **{hub}** (delay rate {delay:.1f}%). "
                    "Reroute overflow orders to nearest low-delay hub. "
                    "Escalate to city ops manager."
                )
            elif severity == "Medium":
                action = (
                    f"Reroute 20–30 % of pending orders from **{hub}** "
                    "to adjacent hubs with available capacity. "
                    "Review SLA breach list for priority re-scheduling."
                )
            else:
                action = (
                    f"Flag **{hub}** for monitoring. Delay rate {delay:.1f}% is above "
                    "threshold. Review traffic/route data for the next 2 hours."
                )

        # ───────────────────────────────────────────────
        # Rule 3 – Routing Inefficiency
        # ───────────────────────────────────────────────
        elif issue == "Routing Inefficiency":
            avg_time = hub_df.loc[hub_df["hub_name"] == hub,
                                   "avg_delivery_time_minutes"].values[0]
            action = (
                f"Average delivery time at **{hub}** is {avg_time:.0f} min. "
                "Cluster deliveries by PIN code to reduce travel distance. "
                "Run dynamic route optimisation for next dispatch cycle."
            )

        else:
            action = "Review hub operations manually."

        recommendations.append({
            "hub_name":           hub,
            "problem":            issue,
            "recommended_action": action,
            "priority":           priority,
        })

    return pd.DataFrame(recommendations).reset_index(drop=True)


def generate_reallocation_plan(hub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match overloaded hubs to donor hubs and compute how many riders to move.

    Overloaded hub : orders_per_rider > 3.0
    Donor hub      : orders_per_rider < 1.5 (spare capacity available)

    The function greedily pairs the most overloaded hub with the most
    under-utilised donor hub, moves as many riders as the donor can spare
    (while keeping donor at ≥ 1.5 opr), and repeats until no more
    beneficial transfers are possible.

    Parameters
    ----------
    hub_df : pd.DataFrame
        Hub snapshot from data_generator.generate_hub_data().

    Returns
    -------
    pd.DataFrame
        Columns:
            from_hub, to_hub, riders_to_move,
            current_opr_receiver, expected_opr_receiver,
            current_opr_donor,    expected_opr_donor
        Returns empty df if no beneficial transfers exist.
    """
    OVERLOAD_THRESH = 3.0
    DONOR_THRESH    = 1.5
    MIN_DONOR_OPR   = 1.5   # keep donor above this after giving riders

    # Working copy with mutable rider counts
    work = hub_df[["hub_name", "orders_incoming", "active_riders"]].copy()
    work["opr"] = work["orders_incoming"] / work["active_riders"].clip(lower=1)

    transfers = []

    for _ in range(len(work)):   # at most one transfer per hub pair
        # Identify current state
        overloaded = work[work["opr"] > OVERLOAD_THRESH].sort_values("opr", ascending=False)
        donors     = work[work["opr"] < DONOR_THRESH].sort_values("opr")

        if overloaded.empty or donors.empty:
            break

        # Pick the most overloaded receiver and the most under-utilised donor
        recv_idx = overloaded.index[0]
        donor_idx = donors.index[0]

        recv  = work.loc[recv_idx]
        donor = work.loc[donor_idx]

        if recv["hub_name"] == donor["hub_name"]:
            break

        recv_orders  = recv["orders_incoming"]
        recv_riders  = recv["active_riders"]
        donor_orders = donor["orders_incoming"]
        donor_riders = donor["active_riders"]

        # How many riders can the donor spare while staying ≥ MIN_DONOR_OPR?
        min_riders_donor = max(1, int(np.ceil(donor_orders / MIN_DONOR_OPR)))
        max_spare        = max(0, int(donor_riders) - min_riders_donor)

        if max_spare < 1:
            break

        # How many riders does the receiver need to drop to ≤ 3.0 opr?
        target_riders_recv = max(1, int(np.ceil(recv_orders / OVERLOAD_THRESH)))
        needed             = max(0, target_riders_recv - int(recv_riders))

        riders_to_move = min(needed, max_spare)
        if riders_to_move < 1:
            break

        new_recv_riders  = int(recv_riders)  + riders_to_move
        new_donor_riders = int(donor_riders) - riders_to_move
        new_recv_opr     = recv_orders  / max(new_recv_riders,  1)
        new_donor_opr    = donor_orders / max(new_donor_riders, 1)

        transfers.append({
            "from_hub":               donor["hub_name"],
            "to_hub":                 recv["hub_name"],
            "riders_to_move":         riders_to_move,
            "current_opr_receiver":   round(float(recv["opr"]),  2),
            "expected_opr_receiver":  round(new_recv_opr,  2),
            "current_opr_donor":      round(float(donor["opr"]), 2),
            "expected_opr_donor":     round(new_donor_opr, 2),
        })

        # Apply the transfer in the working copy so subsequent iterations
        # reflect updated state
        work.loc[recv_idx,  "active_riders"] = new_recv_riders
        work.loc[donor_idx, "active_riders"] = new_donor_riders
        work.loc[recv_idx,  "opr"] = new_recv_opr
        work.loc[donor_idx, "opr"] = new_donor_opr

    if not transfers:
        return pd.DataFrame(columns=[
            "from_hub", "to_hub", "riders_to_move",
            "current_opr_receiver", "expected_opr_receiver",
            "current_opr_donor", "expected_opr_donor",
        ])

    return pd.DataFrame(transfers).reset_index(drop=True)


# ──────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator   import generate_hub_data
    from bottleneck_detector import detect_bottlenecks

    hub_df        = generate_hub_data(bottleneck_hubs=["Bangalore North", "Delhi Central"])
    bottleneck_df = detect_bottlenecks(hub_df)
    recs          = generate_recommendations(bottleneck_df, hub_df)

    for _, r in recs.iterrows():
        print(f"\n[{r['priority']}] {r['hub_name']} — {r['problem']}")
        print(f"  → {r['recommended_action']}")
