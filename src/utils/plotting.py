from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def price_chart(rt_df: pd.DataFrame, da_df: pd.DataFrame):
    fig = go.Figure()
    if not rt_df.empty:
        fig.add_trace(go.Scatter(x=rt_df["timestamp"], y=rt_df["price"], mode="lines", name="Real-time"))
    if not da_df.empty:
        fig.add_trace(go.Scatter(x=da_df["timestamp"], y=da_df["price"], mode="lines", name="Day-ahead", line={"dash": "dot"}))
    fig.update_layout(
        title="Settlement Point Prices",
        height=380,
        margin={"l": 10, "r": 10, "t": 55, "b": 35},
        xaxis_title="Interval time",
        yaxis_title="Price ($/MWh)",
        legend_title="Market",
    )
    return fig


def load_chart(load_df: pd.DataFrame):
    if load_df.empty:
        return go.Figure().update_layout(title="ERCOT Load Forecast", height=320, xaxis_title="Interval time", yaxis_title="Load Forecast (MW)")
    clean = load_df.dropna(subset=["load_mw"]).sort_values("timestamp").copy()
    clean["rolling_baseline_mw"] = clean["load_mw"].rolling(window=min(8, len(clean)), min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean["timestamp"], y=clean["load_mw"], mode="lines", name="ERCOT Load Forecast"))
    fig.add_trace(
        go.Scatter(
            x=clean["timestamp"],
            y=clean["rolling_baseline_mw"],
            mode="lines",
            name="Rolling Baseline",
            line={"dash": "dot"},
        ),
    )
    fig.update_layout(
        title="ERCOT Load Forecast vs Rolling Baseline",
        height=320,
        margin={"l": 10, "r": 10, "t": 55, "b": 35},
        xaxis_title="Interval time",
        yaxis_title="Load Forecast (MW)",
        legend_title="Series",
        hovermode="x unified",
    )
    return fig


def renewable_chart(wind_df: pd.DataFrame, solar_df: pd.DataFrame):
    fig = go.Figure()
    if not wind_df.empty:
        fig.add_trace(go.Scatter(x=wind_df["timestamp"], y=wind_df["wind_mw"], mode="lines", name="Wind"))
    if not solar_df.empty:
        fig.add_trace(go.Scatter(x=solar_df["timestamp"], y=solar_df["solar_mw"], mode="lines", name="Solar"))
    fig.update_layout(
        title="Wind and Solar Generation",
        height=320,
        margin={"l": 10, "r": 10, "t": 55, "b": 35},
        xaxis_title="Interval time",
        yaxis_title="Generation (MW)",
        legend_title="Resource",
    )
    return fig
