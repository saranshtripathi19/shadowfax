# AI Last-Mile Control Tower – Hub Bottleneck Detector (v2)

An industry-grade logistics monitoring dashboard designed for real-time fleet management and operational decision support. Inspired by systems at companies like **Shadowfax**, **Delhivery**, and **Amazon Logistics**.

## 🚀 Key Features

### 1. Hub Health Score (NEW)
A weighted operational index (0-100) calculated using:
*   **SLA Breach %** (35%)
*   **Load Factor** (Orders/Rider) (30%)
*   **Average TAT** (20%)
*   **Rider Shortage** (15%)

### 2. Rider Reallocation Engine (NEW)
Automated matching engine that pairs overloaded hubs with donor hubs. It calculates the exact number of delivery partners to move to balance the network SLA across the city.

### 3. Geospatial Hub Map (NEW)
Interactive Mapbox visualization of hub clusters across major Indian metros (Bangalore, Mumbai, Delhi), color-coded by real-time health status.

### 4. AI Operations Briefing (NEW)
A rule-based narrative panel providing human-readable "AI Operations Briefings" for every hub. It translates complex logistics metrics into Primary Drivers and Suggested Actions.

### 5. Scenario Simulation Tool (NEW)
Interactive "What-If" sandbox to simulate custom shipment volume spikes or partner shortages and observe the immediate downstream impact on SLA and Health Scores.

### 6. Time-Series Monitoring (NEW)
Historical 24-hour monitoring of SLA performance and shipment influx vs delivery partner availability.

## 🏗️ System Architecture

*   **Dash/UI**: Streamlit (Apple-inspired "Glassmorphism" Dark UI)
*   **Visualisations**: Plotly Graph Objects & Mapbox
*   **Logic Modules**: 
    *   `hub_health.py`: Health score calculation.
    *   `ops_explainer.py`: Narrative briefing generation.
    *   `recommendation_engine.py`: Reallocation matching & bottlenecks.
    *   `delay_prediction.py`: Scikit-learn regression for TAT forecasting.
*   **Data**: `data_generator.py` (Synthetic high-entropy logistics data)

## 🏁 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch Dashboard**:
    ```bash
    streamlit run dashboard.py
    ```

---
*Developed for Shadowfax Last-Mile Operations Case Study.*
