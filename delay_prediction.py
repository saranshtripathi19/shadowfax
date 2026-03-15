"""
delay_prediction.py
-------------------
Trains a lightweight ML model on synthetic hub data and predicts
the probability (risk) of delivery delays at each hub.

Model: Linear Regression (scikit-learn)
  - Simple, interpretable, fast to train on small synthetic datasets.
  - Target: delay_rate_percent (continuous)
  - Features: orders_incoming, active_riders, avg_distance_km, orders_processed

The model is trained fresh on every call to predict_delays(), which works
for a prototype because data is synthetic and small.
In production this would be replaced by a pre-trained or incrementally
updated model persisted to disk.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────
# Feature columns used during training / inference
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "orders_incoming",
    "active_riders",
    "avg_distance_km",
    "orders_processed",
]
TARGET_COL = "delay_rate_percent"


def _build_training_data(seed: int = 42) -> pd.DataFrame:
    """
    Create a larger synthetic training dataset (500 samples) so the
    regression has enough variance to learn a meaningful surface.

    We deliberately do NOT import data_generator here to keep this module
    self-contained.  The training distribution matches the generator's ranges.
    """
    rng = np.random.default_rng(seed)
    n = 500

    orders_incoming   = rng.integers(50, 400, size=n).astype(float)
    active_riders     = rng.integers(5,  100, size=n).astype(float)
    avg_distance_km   = rng.uniform(2,   15,  size=n)
    orders_processed  = np.clip(orders_incoming * rng.uniform(0.5, 1.0, size=n),
                                0, orders_incoming)

    # Delay rate is driven by:
    #   → orders per rider (congestion)
    #   → distance (longer routes → more delays)
    #   → some random noise
    orders_per_rider = orders_incoming / np.maximum(active_riders, 1)
    delay_rate = (
        2.5 * orders_per_rider
        + 1.2 * avg_distance_km
        - 0.05 * orders_processed
        + rng.normal(0, 3, size=n)          # measurement noise
    ).clip(0, 100)                           # keep in [0, 100] percent

    return pd.DataFrame({
        "orders_incoming":  orders_incoming,
        "active_riders":    active_riders,
        "avg_distance_km":  avg_distance_km,
        "orders_processed": orders_processed,
        TARGET_COL:         delay_rate,
    })


def predict_delays(hub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict delivery-delay risk for each hub in *hub_df*.

    The function:
      1. Builds and trains a LinearRegression on synthetic training data.
      2. Scales features with MinMaxScaler for numerical stability.
      3. Predicts delay_rate on the live hub snapshot.
      4. Converts the predicted rate to a clamped [0, 100] % risk score.

    Parameters
    ----------
    hub_df : pd.DataFrame
        Hub snapshot from data_generator.generate_hub_data().
        Must contain all columns in FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        Columns: hub_name, predicted_delay_pct, delay_risk_label
          predicted_delay_pct  – raw regression output clipped to [0, 100]
          delay_risk_label     – "Low" / "Medium" / "High"
    """
    # ── Train ──
    train_df = _build_training_data()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    y_train = train_df[TARGET_COL].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ── Predict on live hub snapshot ──
    X_live = scaler.transform(hub_df[FEATURE_COLS])
    preds  = model.predict(X_live)
    preds  = np.clip(preds, 0, 100)  # keep predictions physically meaningful

    # ── Assign risk label ──
    def _label(p: float) -> str:
        if p < 25:
            return "Low"
        elif p < 45:
            return "Medium"
        else:
            return "High"

    result = pd.DataFrame({
        "hub_name":            hub_df["hub_name"].values,
        "predicted_delay_pct": preds.round(1),
        "delay_risk_label":    [_label(p) for p in preds],
    })

    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_hub_data
    df = generate_hub_data(bottleneck_hubs=["Bangalore North"])
    result = predict_delays(df)
    print(result.to_string(index=False))
