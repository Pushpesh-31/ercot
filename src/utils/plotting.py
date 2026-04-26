from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def price_load_overlay_chart(rt_df: pd.DataFrame, da_df: pd.DataFrame, load_df: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not rt_df.empty:
        fig.add_trace(
            go.Scatter(x=rt_df["timestamp"], y=rt_df["price"], mode="lines", name="Real-Time Price"),
            secondary_y=False,
        )
    if not da_df.empty:
        fig.add_trace(
            go.Scatter(x=da_df["timestamp"], y=da_df["price"], mode="lines", name="Day-Ahead Price", line={"dash": "dot"}),
            secondary_y=False,
        )
    if not load_df.empty:
        fig.add_trace(
            go.Scatter(x=load_df["timestamp"], y=load_df["load_mw"], mode="lines", name="ERCOT Load Forecast", line={"dash": "dash"}),
            secondary_y=True,
        )
    fig.update_layout(
        title="Prices Overlaid With ERCOT Load Forecast",
        height=420,
        margin={"l": 10, "r": 10, "t": 55, "b": 35},
        xaxis_title="Interval time",
        legend_title="Series",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="Load Forecast (MW)", secondary_y=True)
    return fig


def load_chart(load_df: pd.DataFrame):
    if load_df.empty:
        return go.Figure().update_layout(title="ERCOT Load", height=320, xaxis_title="Interval time", yaxis_title="Load (MW)")
    return px.line(load_df, x="timestamp", y="load_mw", labels={"load_mw": "Load (MW)", "timestamp": "Interval time"}).update_layout(
        title="ERCOT Load",
        height=320,
        margin={"l": 10, "r": 10, "t": 55, "b": 35},
    )


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
