# ERCOT Grid-to-Price Intelligence Agent

A Streamlit app that connects ERCOT physical grid conditions to real-time and day-ahead price movement, then produces an evidence-backed market briefing for an industrial consumer, generator / asset owner, or market observer.

This is not a trading bot and does not provide financial or trading advice. Human review is required.

## Why ERCOT

ERCOT publishes detailed public power-market data for the Texas grid, including settlement point prices, load forecasts, and renewable generation reports. That makes it a useful market for connecting observed grid changes to price behavior using transparent public data.

## Data Sources

The app uses ERCOT public API data only:

- Real-Time Settlement Point Prices: `NP6-905-CD`, endpoint `/np6-905-cd/spp_node_zone_hub`
- Day-Ahead Settlement Point Prices: `NP4-190-CD`, endpoint `/np4-190-cd/dam_stlmnt_pnt_prices`
- Load forecast / demand: `NP3-565-CD`, endpoint `/np3-565-cd/lf_by_model_weather_zone`
- Wind generation actual and forecast: `NP4-732-CD`, endpoint `/np4-732-cd/wpp_hrly_avrg_actl_fcast`
- Solar generation actual and forecast: `NP4-737-CD`, endpoint `/np4-737-cd/spp_hrly_avrg_actl_fcast`

The implementation uses `gridstatus` as a helper around the ERCOT public API and `requests`-based authentication.

## ERCOT API Credentials

Register at the ERCOT API Explorer:

https://apiexplorer.ercot.com/

Create an account, subscribe to the ERCOT Public API product, and copy the subscription key. The app expects:

```bash
ERCOT_USERNAME=
ERCOT_PASSWORD=
ERCOT_SUBSCRIPTION_KEY=
```

For local development, copy `.env.example` to `.env` and fill in the values. Do not commit `.env`.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
pytest
```

## Refresh Policy

V1 refreshes data hourly using Streamlit caching with a 60-minute TTL. The sidebar includes a `Refresh now` button that clears the cache and pulls fresh ERCOT data. The last successful pull metadata is stored in `data/cache/`.

ERCOT real-time settlement point prices are generated every 15 minutes, so a later version can reduce the TTL to support 15-minute refreshes.

## Streamlit Community Cloud

Deploy the repo as a Streamlit app with `app.py` as the entry point. Add the following secrets in Streamlit Community Cloud:

```toml
ERCOT_USERNAME = "your_username"
ERCOT_PASSWORD = "your_password"
ERCOT_SUBSCRIPTION_KEY = "your_subscription_key"
```

No `.env` file should be committed or uploaded.

## How the Agents Work

The app runs a deterministic four-step workflow:

1. Grid Signal Agent detects load, wind, solar, and possible supply tightening signals.
2. Price Impact Agent detects real-time price spikes, drops, volatility, negative prices, and real-time versus day-ahead spreads.
3. Reasoning Agent connects observed grid signals to observed price signals with confidence scoring.
4. Exposure & Action Agent translates signals into non-prescriptive persona-specific considerations.

All recommendations are linked to detected signals. Missing data is reported as `insufficient data`.

## Limitations

V1 does not model all ERCOT market drivers. Transmission constraints, outages, ancillary service scarcity, congestion, and local topology can materially affect prices. The app uses rule-based interpretation and observational data, so it should not be treated as proof of causality.

This is not financial or trading advice. Human review required.
