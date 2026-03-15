"""
dashboard.py  ·  AI Last-Mile Logistics Control Tower  ·  v2
-------------------------------------------------------------
Apple-inspired, logistics industry-grade Streamlit dashboard.

Sections
--------
 ①  Header + Live KPIs
 ②  Hub Health Score Overview
 ③  Logistics Hub Map
 ④  Hub Metrics Table
 ⑤  Bottleneck Alerts
 ⑥  Rider Reallocation Plan
 ⑦  AI Recommendations
 ⑧  AI Ops Explanation Panel
 ⑨  Delay Prediction
 ⑩  Time-Series Monitoring
 ⑪  Scenario Simulator

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Local module imports ──────────────────────────────────────────────────────
from data_generator        import generate_hub_data, generate_timeseries_data
from bottleneck_detector   import detect_bottlenecks
from delay_prediction      import predict_delays
from recommendation_engine import generate_recommendations, generate_reallocation_plan
from hub_health            import calculate_hub_health_score
from hub_map               import build_hub_map
from ops_explainer         import generate_all_explanations

# ─────────────────────────────────────────────────────────────────────────────
# ①  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Control Tower · Shadowfax",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Apple-inspired dark system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding: 1.5rem 2rem 3rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

/* ── Section label ── */
.section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    color: #64748b; text-transform: uppercase; margin-bottom: 10px;
}

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.kpi-label  { font-size: 11px; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
.kpi-value  { font-size: 2.1rem; font-weight: 700; color: #f8fafc; margin: 4px 0 2px; line-height: 1.1; }
.kpi-sub    { font-size: 12px; color: #475569; }
.kpi-green  { color: #34d399; } .kpi-red   { color: #f87171; } .kpi-amber { color: #fbbf24; }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 28px 0 24px;
}

/* ── Health badge ── */
.badge {
    display: inline-block;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 100px; letter-spacing: 0.04em;
}
.badge-healthy  { background: #052e16; color: #34d399; }
.badge-warning  { background: #451a03; color: #fbbf24; }
.badge-critical { background: #450a0a; color: #f87171; }

/* ── Alert card ── */
.alert-card {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    border-left: 4px solid;
}
.alert-high   { background: #1a0a0a; border-color: #ef4444; }
.alert-medium { background: #1a0f00; border-color: #f59e0b; }
.alert-low    { background: #031a0e; border-color: #22c55e; }
.alert-title  { font-weight: 600; font-size: 0.9rem; color: #f1f5f9; margin-bottom: 3px; }
.alert-meta   { font-size: 0.78rem; color: #94a3b8; }

/* ── Recommendation card ── */
.rec-card {
    background: #0f1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.rec-hub    { font-size: 0.95rem; font-weight: 700; color: #38bdf8; }
.rec-badge  { font-size: 0.75rem; color: #64748b; margin-bottom: 8px; }
.rec-action { font-size: 0.85rem; color: #cbd5e1; line-height: 1.65; }

/* ── Realloc table ── */
.realloc-header {
    font-size: 11px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.12em; padding: 6px 0; border-bottom: 1px solid #1e293b;
    display: grid; grid-template-columns: 2fr 2fr 1fr 1.5fr 1.5fr;
    margin-bottom: 4px;
}
.realloc-row {
    font-size: 0.82rem; color: #e2e8f0; padding: 10px 0;
    border-bottom: 1px solid #0f1a2e;
    display: grid; grid-template-columns: 2fr 2fr 1fr 1.5fr 1.5fr;
    align-items: center;
}
.arrow { color: #38bdf8 }

/* ── Expander override ── */
details summary { font-weight: 600; color: #94a3b8 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0f1a2e 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown { color: #64748b; }

/* ── Plotly chart background ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Metric cards (native Streamlit fallback) ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 18px 20px;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem !important; }

/* ── Simulator panel ── */
.sim-card {
    background: #0f1a2e; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
}
.sim-hub { font-size: 0.9rem; font-weight: 700; color: #7dd3fc; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chart theme helper
# ─────────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor="#080f1e",
    paper_bgcolor="#0a1628",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(t=48, b=10, l=10, r=10),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", tickcolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)

STATUS_COLOR = {"Healthy": "#34d399", "Warning": "#fbbf24", "Critical": "#ef4444"}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — scenario controls
# ─────────────────────────────────────────────────────────────────────────────
ALL_HUBS = [
    "Bangalore North", "Bangalore East", "Bangalore South", "Bangalore West",
    "Delhi Central", "Delhi South", "Delhi East",
    "Mumbai North", "Mumbai South", "Hyderabad Central",
]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 6px'>
        <span style='font-size:2rem'>🚚</span>
        <div style='font-size:0.7rem; color:#38bdf8; letter-spacing:0.2em; text-transform:uppercase; font-weight:600; margin-top:4px'>
            Control Tower
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Stress-Test Hubs</div>", unsafe_allow_html=True)

    selected_bottleneck_hubs = st.multiselect(
        "Inject bottleneck into:",
        options=ALL_HUBS,
        default=["Bangalore North", "Delhi Central"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Reproducibility</div>", unsafe_allow_html=True)
    random_seed = st.number_input("Random seed  (0 = random)", 0, 9999, 42, label_visibility="visible")
    seed_val = int(random_seed) if random_seed != 0 else None

    st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:12px 0'>", unsafe_allow_html=True)
    st.caption("© 2025 Shadowfax AI Control Tower")


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline (cached in session_state)
# ─────────────────────────────────────────────────────────────────────────────
def _run_pipeline(bottleneck_hubs, seed):
    hub_df        = generate_hub_data(bottleneck_hubs=bottleneck_hubs or None, seed=seed)
    bottleneck_df = detect_bottlenecks(hub_df)
    prediction_df = predict_delays(hub_df)
    recs_df       = generate_recommendations(bottleneck_df, hub_df)
    realloc_df    = generate_reallocation_plan(hub_df)
    health_df     = calculate_hub_health_score(hub_df)
    map_fig       = build_hub_map(hub_df, health_df)
    explanations  = generate_all_explanations(hub_df, health_df)
    ts_df         = generate_timeseries_data(
        hours=24, bottleneck_hubs=bottleneck_hubs or None, seed=seed
    )
    return hub_df, bottleneck_df, prediction_df, recs_df, realloc_df, health_df, map_fig, explanations, ts_df


# ─────────────────────────────────────────────────────────────────────────────
# ①  HEADER + REFRESH
# ─────────────────────────────────────────────────────────────────────────────
col_h, col_ts, col_btn = st.columns([4, 2, 1])
with col_h:
    st.markdown("""
    <div style='line-height:1.2'>
        <div style='font-size:1.75rem; font-weight:700; color:#f8fafc; letter-spacing:-0.02em'>
            AI Last-Mile Control Tower
        </div>
        <div style='font-size:0.8rem; color:#475569; margin-top:4px'>
            Real-time hub monitoring · Bottleneck detection · AI recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_ts:
    st.markdown(
        f"<div style='font-size:0.78rem; color:#475569; margin-top:18px; text-align:right'>"
        f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
        unsafe_allow_html=True
    )
with col_btn:
    st.write("")
    refresh = st.button("⟳  Refresh", use_container_width=True)

if "pipeline_data" not in st.session_state or refresh:
    with st.spinner("Running AI pipeline…"):
        st.session_state["pipeline_data"] = _run_pipeline(selected_bottleneck_hubs, seed_val)

(hub_df, bottleneck_df, prediction_df, recs_df, realloc_df,
 health_df, map_fig, explanations, ts_df) = st.session_state["pipeline_data"]

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Live network KPIs
# ─────────────────────────────────────────────────────────────────────────────
total_shipments   = hub_df["orders_incoming"].sum()
total_partners    = hub_df["active_riders"].sum()
avg_tat           = hub_df["avg_delivery_time_minutes"].mean()
avg_sla_breach    = hub_df["delay_rate_percent"].mean()
num_alerts        = len(bottleneck_df)
critical_count    = len(health_df[health_df["health_status"] == "Critical"])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Shipment Volume</div>
        <div class='kpi-value'>{total_shipments:,}</div>
        <div class='kpi-sub'>Across all 10 hubs</div>
    </div>""", unsafe_allow_html=True)
with k2:
    clr = "kpi-red" if total_partners < 350 else "kpi-green"
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Active Delivery Partners</div>
        <div class='kpi-value {clr}'>{total_partners:,}</div>
        <div class='kpi-sub'>Network-wide fleet</div>
    </div>""", unsafe_allow_html=True)
with k3:
    clr = "kpi-red" if avg_tat > 45 else ("kpi-amber" if avg_tat > 38 else "kpi-green")
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Avg Delivery TAT</div>
        <div class='kpi-value {clr}'>{avg_tat:.1f} <span style='font-size:1rem'>min</span></div>
        <div class='kpi-sub'>SLA target: &lt; 45 min</div>
    </div>""", unsafe_allow_html=True)
with k4:
    clr = "kpi-red" if avg_sla_breach > 20 else ("kpi-amber" if avg_sla_breach > 12 else "kpi-green")
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>SLA Breach %</div>
        <div class='kpi-value {clr}'>{avg_sla_breach:.1f}%</div>
        <div class='kpi-sub'>{num_alerts} active alert(s) · {critical_count} critical</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ②  HUB HEALTH SCORE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Hub Health Score</div>", unsafe_allow_html=True)

col_hbar, col_htable = st.columns([3, 2])

with col_hbar:
    health_sorted = health_df.sort_values("health_score")
    bar_colors    = [STATUS_COLOR[s] for s in health_sorted["health_status"]]
    fig_health = go.Figure(go.Bar(
        x=health_sorted["health_score"],
        y=health_sorted["hub_name"],
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.85),
        text=health_sorted["health_score"].astype(str),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_health.add_vline(x=40, line=dict(color="#ef4444", dash="dot", width=1))
    fig_health.add_vline(x=70, line=dict(color="#fbbf24", dash="dot", width=1))
    fig_health.update_layout(**CHART_LAYOUT)
    fig_health.update_layout(
        title=dict(text="Hub Health Score (0 = Critical, 100 = Healthy)", font=dict(size=13, color="#94a3b8")),
        xaxis=dict(range=[0, 110], **CHART_LAYOUT["xaxis"]),
        height=360,
    )
    st.plotly_chart(fig_health, use_container_width=True)

with col_htable:
    st.markdown("<div style='padding-top:38px'></div>", unsafe_allow_html=True)
    for _, row in health_df.sort_values("health_score").iterrows():
        badge_cls = {"Healthy": "badge-healthy", "Warning": "badge-warning", "Critical": "badge-critical"}.get(row["health_status"], "")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:9px 0;border-bottom:1px solid #1e293b'>"
            f"<span style='font-size:0.83rem;color:#e2e8f0'>{row['hub_name']}</span>"
            f"<div>"
            f"<span style='font-size:0.83rem;color:#94a3b8;margin-right:12px'>{row['health_score']:.0f}</span>"
            f"<span class='badge {badge_cls}'>{row['health_status']}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ③  LOGISTICS HUB MAP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Logistics Hub Map</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:0.8rem;color:#475569;margin-bottom:10px'>Dot size reflects criticality · Colour indicates health status</div>", unsafe_allow_html=True)
st.plotly_chart(map_fig, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ④  HUB METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Hub Metrics — Live Snapshot</div>", unsafe_allow_html=True)

display_df = hub_df[[
    "hub_name", "city", "orders_incoming", "orders_processed",
    "active_riders", "avg_delivery_time_minutes", "delay_rate_percent", "avg_distance_km"
]].copy()
display_df.columns = [
    "Hub", "City", "Shipment Vol", "Orders Out",
    "Delivery Partners", "Avg TAT (min)", "SLA Breach %", "Avg Distance (km)"
]

def _style_sla(val):
    if val > 30:  return "color:#f87171;font-weight:700"
    elif val > 20: return "color:#fbbf24;font-weight:600"
    else:          return "color:#34d399"

st.dataframe(
    display_df.style.applymap(_style_sla, subset=["SLA Breach %"]),
    use_container_width=True, hide_index=True,
)

fig_ov = px.bar(
    hub_df, x="hub_name",
    y=["orders_incoming", "orders_processed"],
    barmode="group",
    labels={"hub_name": "Hub", "value": "Shipments", "variable": ""},
    color_discrete_map={"orders_incoming": "#38bdf8", "orders_processed": "#34d399"},
    template="plotly_dark",
)
fig_ov.update_layout(
    **CHART_LAYOUT,
    title=dict(text="Shipment Volume vs Processed — All Hubs", font=dict(size=13, color="#94a3b8")),
    xaxis_tickangle=-30, legend_title_text="",
)
st.plotly_chart(fig_ov, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑤  BOTTLENECK ALERTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Bottleneck Alerts</div>", unsafe_allow_html=True)

if bottleneck_df.empty:
    st.success("✅  No bottlenecks detected — network operating normally.")
else:
    sev_count = bottleneck_df["severity"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", sev_count.get("High",   0))
    c2.metric("🟡 Warning",  sev_count.get("Medium", 0))
    c3.metric("🟢 Low",      sev_count.get("Low",    0))

    for _, row in bottleneck_df.iterrows():
        css = {"High": "alert-high", "Medium": "alert-medium", "Low": "alert-low"}.get(row["severity"], "alert-low")
        icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(row["severity"], "⚪")
        st.markdown(f"""
        <div class='alert-card {css}'>
            <div class='alert-title'>{icon}  {row['hub_name']}  ·  {row['issue_type']}</div>
            <div class='alert-meta'>
                Severity: <strong>{row['severity']}</strong> &nbsp;|&nbsp;
                Orders/Partner: <strong>{row['orders_per_rider']:.1f}</strong> &nbsp;|&nbsp;
                SLA Breach: <strong>{row['delay_rate']}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    sev_map = {"Low": 1, "Medium": 2, "High": 3}
    heat_df = bottleneck_df[["hub_name","issue_type","severity"]].copy()
    heat_df["sev_num"] = heat_df["severity"].map(sev_map)
    fig_heat = px.bar(
        heat_df, x="hub_name", y="sev_num", color="issue_type",
        barmode="group",
        labels={"hub_name":"Hub","sev_num":"Severity","issue_type":"Issue"},
        color_discrete_sequence=["#38bdf8","#f59e0b","#a78bfa"],
    )
    fig_heat.update_layout(**CHART_LAYOUT)
    fig_heat.update_layout(
        title=dict(text="Bottleneck Severity by Hub", font=dict(size=13, color="#94a3b8")),
        xaxis_tickangle=-30,
        yaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"], **CHART_LAYOUT["yaxis"]),
        legend_title_text="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑥  RIDER REALLOCATION PLAN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Rider Reallocation Engine</div>", unsafe_allow_html=True)

if realloc_df.empty:
    st.info("⚖️  Load is balanced — no rider transfers required.")
else:
    st.markdown(
        "<div style='font-size:0.82rem;color:#64748b;margin-bottom:12px'>"
        "Greedy matching of overloaded hubs (orders/partner > 3.0) to donor hubs (< 1.5)</div>",
        unsafe_allow_html=True
    )
    for _, r in realloc_df.iterrows():
        opr_delta = r["current_opr_receiver"] - r["expected_opr_receiver"]
        st.markdown(f"""
        <div class='rec-card' style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
            <div>
                <span style='font-size:0.82rem;color:#64748b'>From</span>
                <span style='font-size:0.88rem;font-weight:600;color:#34d399;margin:0 10px'>{r['from_hub']}</span>
                <span style='color:#38bdf8;font-size:1rem'>→</span>
                <span style='font-size:0.88rem;font-weight:600;color:#f87171;margin:0 10px'>{r['to_hub']}</span>
                <span style='font-size:0.82rem;color:#64748b'>To</span>
            </div>
            <div style='display:flex;gap:24px;flex-wrap:wrap'>
                <div style='text-align:center'>
                    <div style='font-size:1.4rem;font-weight:700;color:#38bdf8'>{r['riders_to_move']}</div>
                    <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em'>Partners to move</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:0.88rem;color:#94a3b8'>{r['current_opr_receiver']:.1f} → <strong style="color:#34d399">{r['expected_opr_receiver']:.1f}</strong></div>
                    <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em'>Receiver OPR</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:0.88rem;color:#94a3b8'>{r['current_opr_donor']:.1f} → <strong style="color:#fbbf24">{r['expected_opr_donor']:.1f}</strong></div>
                    <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em'>Donor OPR</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑦  AI RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>AI Recommendations</div>", unsafe_allow_html=True)

if recs_df.empty:
    st.info("✅  All hubs operating within normal parameters — no recommendations.")
else:
    for _, rec in recs_df.iterrows():
        st.markdown(f"""
        <div class='rec-card'>
            <div class='rec-hub'>{rec['hub_name']}</div>
            <div class='rec-badge'>{rec['priority']}  ·  {rec['problem']}</div>
            <div class='rec-action'>{rec['recommended_action']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑧  AI OPS EXPLANATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>AI Operations Briefing</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.8rem;color:#475569;margin-bottom:10px'>"
    "Rule-based narrative analysis for each hub · Expand to read</div>",
    unsafe_allow_html=True
)

# Show critical/warning hubs first in the expander list
exp_order = health_df.sort_values("health_score")["hub_name"].tolist()

for hub_name in exp_order:
    status = health_df.loc[health_df["hub_name"] == hub_name, "health_status"].values[0]
    score  = health_df.loc[health_df["hub_name"] == hub_name, "health_score"].values[0]
    icon   = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}.get(status, "⚪")
    with st.expander(f"{icon}  {hub_name}  —  Health {score:.0f} / 100"):
        st.markdown(explanations.get(hub_name, "_No explanation available._"))

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑨  DELAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Delay Prediction  ·  ML Model</div>", unsafe_allow_html=True)
st.caption("Linear regression trained on synthetic distribution")

pred_display = prediction_df.copy()
pred_display["actual_delay_%"] = hub_df["delay_rate_percent"].values
pred_display["color"] = pred_display["delay_risk_label"].map(
    {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#34d399"}
)

fig_pred = go.Figure()
fig_pred.add_trace(go.Bar(
    x=pred_display["hub_name"],
    y=pred_display["predicted_delay_pct"],
    name="Predicted SLA Breach %",
    marker_color=pred_display["color"],
    opacity=0.8,
    text=pred_display["predicted_delay_pct"].round(1).astype(str) + "%",
    textposition="outside",
    textfont=dict(size=10),
))
fig_pred.add_trace(go.Scatter(
    x=pred_display["hub_name"],
    y=pred_display["actual_delay_%"],
    name="Actual SLA Breach %",
    mode="lines+markers",
    line=dict(color="#38bdf8", width=2, dash="dot"),
    marker=dict(size=7, color="#38bdf8"),
))
fig_pred.update_layout(**CHART_LAYOUT)
fig_pred.update_layout(
    title=dict(text="Predicted vs Actual SLA Breach % per Hub", font=dict(size=13, color="#94a3b8")),
    xaxis_tickangle=-30,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)
st.plotly_chart(fig_pred, use_container_width=True)

# Risk table
risk_table = pred_display[["hub_name","predicted_delay_pct","delay_risk_label","actual_delay_%"]].copy()
risk_table.columns = ["Hub","Predicted SLA Breach %","Risk Level","Actual SLA Breach %"]
def _hl_risk(val):
    return {"High": "background:#1a0a0a", "Medium": "background:#1a0f00", "Low": "background:#031a0e"}.get(val, "")
st.dataframe(risk_table.style.applymap(_hl_risk, subset=["Risk Level"]), use_container_width=True, hide_index=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑩  TIME-SERIES MONITORING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Time-Series Monitoring  ·  Last 24 Hours</div>", unsafe_allow_html=True)

ts_hub_filter = st.multiselect(
    "Filter hubs for trend charts:",
    options=ALL_HUBS,
    default=["Bangalore North", "Delhi Central", "Mumbai South"],
    key="ts_filter",
)

ts_filtered = ts_df[ts_df["hub_name"].isin(ts_hub_filter)] if ts_hub_filter else ts_df

# SLA Breach % trend
fig_delay_ts = px.line(
    ts_filtered.sort_values("timestamp"),
    x="timestamp", y="delay_rate_percent",
    color="hub_name",
    labels={"timestamp":"Time","delay_rate_percent":"SLA Breach %","hub_name":"Hub"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_delay_ts.add_hline(y=20, line=dict(color="#ef4444", dash="dot", width=1),
                       annotation_text="20% threshold", annotation_font_color="#ef4444")
fig_delay_ts.update_layout(
    **CHART_LAYOUT,
    title=dict(text="SLA Breach % Trend — 24-Hour Window", font=dict(size=13, color="#94a3b8")),
    legend_title_text="",
)
st.plotly_chart(fig_delay_ts, use_container_width=True)

# Orders vs Riders trend
fig_opr_ts = go.Figure()
colors_line = ["#38bdf8","#a78bfa","#34d399","#fb923c","#f472b6"]
for i, hub in enumerate(ts_hub_filter or ALL_HUBS[:3]):
    sub = ts_filtered[ts_filtered["hub_name"] == hub].sort_values("timestamp")
    clr = colors_line[i % len(colors_line)]
    fig_opr_ts.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["orders_incoming"],
        name=f"{hub} — Shipments", line=dict(color=clr, width=2),
    ))
    fig_opr_ts.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["active_riders"],
        name=f"{hub} — Delivery Partners",
        line=dict(color=clr, width=1.5, dash="dot"),
    ))
fig_opr_ts.update_layout(**CHART_LAYOUT)
fig_opr_ts.update_layout(
    title=dict(text="Shipment Volume vs Delivery Partners — 24-Hour Trend", font=dict(size=13, color="#94a3b8")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10),
                orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_opr_ts, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ⑪  SCENARIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Scenario Simulator</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.8rem;color:#475569;margin-bottom:16px'>"
    "Simulate custom conditions for a single hub and instantly see recalculated health, "
    "bottlenecks, and delay predictions.</div>",
    unsafe_allow_html=True
)

sim_hub = st.selectbox("Select hub to simulate:", ALL_HUBS, key="sim_hub")

sim_col1, sim_col2 = st.columns(2)
# Seed defaults from current live data
live_row    = hub_df[hub_df["hub_name"] == sim_hub].iloc[0]
default_ord = int(live_row["orders_incoming"])
default_rid = int(live_row["active_riders"])

with sim_col1:
    sim_orders  = st.slider("Shipment Volume", 10, 500, default_ord, step=5, key="sim_orders")
with sim_col2:
    sim_riders  = st.slider("Active Delivery Partners", 1, 150, default_rid, step=1, key="sim_riders")

# --- Build a synthetic one-hub dataframe with the simulated values ---
sim_row = live_row.copy()
sim_row["orders_incoming"]  = sim_orders
sim_row["active_riders"]    = sim_riders
sim_row["orders_processed"] = min(sim_orders, sim_riders * 4)
sim_df = pd.DataFrame([sim_row])

sim_opr         = sim_orders / max(sim_riders, 1)
sim_health_df   = calculate_hub_health_score(sim_df)
sim_health_score= sim_health_df["health_score"].values[0]
sim_health_stat = sim_health_df["health_status"].values[0]
sim_bottleneck  = detect_bottlenecks(sim_df)
sim_pred        = predict_delays(sim_df)
sim_pred_pct    = sim_pred["predicted_delay_pct"].values[0]
sim_risk        = sim_pred["delay_risk_label"].values[0]
sim_expl        = generate_all_explanations(sim_df, sim_health_df)

# Display results
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Orders / Partner", f"{sim_opr:.2f}", delta=f"{sim_opr - live_row['orders_incoming']/max(live_row['active_riders'],1):.2f}",
              delta_color="inverse")
with s2:
    badge_c = {"Healthy":"#34d399","Warning":"#fbbf24","Critical":"#ef4444"}.get(sim_health_stat, "#94a3b8")
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Health Score</div>"
        f"<div class='kpi-value' style='color:{badge_c}'>{sim_health_score:.0f}</div>"
        f"<div class='kpi-sub'>{sim_health_stat}</div></div>",
        unsafe_allow_html=True
    )
with s3:
    risk_c = {"High":"#ef4444","Medium":"#f59e0b","Low":"#34d399"}.get(sim_risk, "#94a3b8")
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Predicted SLA Breach</div>"
        f"<div class='kpi-value' style='color:{risk_c}'>{sim_pred_pct:.1f}%</div>"
        f"<div class='kpi-sub'>Risk: {sim_risk}</div></div>",
        unsafe_allow_html=True
    )
with s4:
    issues_found = len(sim_bottleneck)
    issue_color  = "#ef4444" if issues_found else "#34d399"
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Bottleneck Issues</div>"
        f"<div class='kpi-value' style='color:{issue_color}'>{issues_found}</div>"
        f"<div class='kpi-sub'>{'Active' if issues_found else 'None found'}</div></div>",
        unsafe_allow_html=True
    )

if not sim_bottleneck.empty:
    st.warning("**Bottlenecks detected in simulation:**")
    for _, brow in sim_bottleneck.iterrows():
        st.markdown(f"- **{brow['issue_type']}** (Severity: {brow['severity']})")

with st.expander("📋  AI Ops Briefing for simulated scenario"):
    st.markdown(sim_expl.get(sim_hub, "—"))

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.caption("🚚  AI Last-Mile Control Tower v2  ·  Built with Streamlit · scikit-learn · Plotly")
