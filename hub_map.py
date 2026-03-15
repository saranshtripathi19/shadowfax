"""
hub_map.py
----------
Renders an interactive Plotly Scattermapbox showing all delivery hubs
coloured by their health status.

No external API key is required — uses Plotly's built-in open-street-map
tile provider (available since Plotly 5.x).

Hub coordinates are real city-level lat/lon values for the 10 Indian hubs.
"""

import pandas as pd
import plotly.graph_objects as go


# ── Hard-coded coordinates (lat, lon) for each hub ────────────────────────
# Using real neighbourhood-level coordinates for Indian cities
HUB_COORDS = {
    "Bangalore North":   (13.0827, 77.5877),   # Hebbal area
    "Bangalore East":    (12.9716, 77.6412),   # Indiranagar area
    "Bangalore South":   (12.9141, 77.5990),   # Jayanagar area
    "Bangalore West":    (12.9784, 77.5408),   # Rajajinagar area
    "Delhi Central":     (28.6448, 77.2167),   # Connaught Place area
    "Delhi South":       (28.5355, 77.2100),   # Lajpat Nagar area
    "Delhi East":        (28.6280, 77.2780),   # Preet Vihar area
    "Mumbai North":      (19.2183, 72.9781),   # Dahisar area
    "Mumbai South":      (18.9388, 72.8354),   # Byculla area
    "Hyderabad Central": (17.3850, 78.4867),   # Secunderabad area
}

# ── Colour palette mapped to health status ────────────────────────────────
STATUS_COLOUR = {
    "Healthy":  "#22c55e",   # green
    "Warning":  "#f59e0b",   # amber
    "Critical": "#ef4444",   # red
}

STATUS_SYMBOL = {
    "Healthy":  "circle",
    "Warning":  "circle",
    "Critical": "circle",
}


def build_hub_map(
    hub_df: pd.DataFrame,
    health_df: pd.DataFrame,
) -> go.Figure:
    """
    Build an interactive hub map coloured by health status.

    Parameters
    ----------
    hub_df : pd.DataFrame
        Hub snapshot (from data_generator.generate_hub_data()).
    health_df : pd.DataFrame
        Health scores (from hub_health.calculate_hub_health_score()).

    Returns
    -------
    go.Figure
        Plotly figure ready to embed in Streamlit via st.plotly_chart().
    """
    # Merge to get health info alongside operational metrics
    merged = hub_df.merge(health_df, on="hub_name", how="left")

    # Attach coordinates
    merged["lat"] = merged["hub_name"].map(lambda h: HUB_COORDS.get(h, (20.0, 78.0))[0])
    merged["lon"] = merged["hub_name"].map(lambda h: HUB_COORDS.get(h, (20.0, 78.0))[1])
    merged["colour"] = merged["health_status"].map(STATUS_COLOUR).fillna("#94a3b8")

    # Compute orders_per_rider for tooltip
    merged["orders_per_rider"] = (
        merged["orders_incoming"] / merged["active_riders"].clip(lower=1)
    ).round(2)

    # Build one trace per health status (so legend groups them correctly)
    fig = go.Figure()

    for status in ["Healthy", "Warning", "Critical"]:
        sub = merged[merged["health_status"] == status]
        if sub.empty:
            continue

        hover_text = [
            f"<b>{row.hub_name}</b><br>"
            f"Health: {row.health_score:.0f} ({row.health_status})<br>"
            f"SLA Breach: {row.delay_rate_percent:.1f}%<br>"
            f"Shipment Vol: {row.orders_incoming}<br>"
            f"Delivery Partners: {row.active_riders}<br>"
            f"Orders/Rider: {row.orders_per_rider:.1f}<br>"
            f"Avg TAT: {row.avg_delivery_time_minutes:.0f} min"
            for row in sub.itertuples()
        ]

        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"],
            lon=sub["lon"],
            mode="markers+text",
            marker=dict(
                size=20 + (100 - sub["health_score"]) * 0.3,  # bigger = more critical
                color=STATUS_COLOUR[status],
                opacity=0.85,
            ),
            text=sub["hub_name"].str.replace(" ", "<br>"),
            textposition="top center",
            textfont=dict(size=10, color="#f1f5f9"),
            hovertext=hover_text,
            hoverinfo="text",
            name=status,
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=18.0, lon=77.5),   # centre of India
            zoom=4.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        paper_bgcolor="#0f172a",
        legend=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#f1f5f9"),
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
        ),
    )

    return fig


# ── Quick smoke-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_hub_data
    from hub_health import calculate_hub_health_score

    hub_df    = generate_hub_data(bottleneck_hubs=["Bangalore North"])
    health_df = calculate_hub_health_score(hub_df)
    fig       = build_hub_map(hub_df, health_df)
    fig.show()
    print("Map built successfully.")
