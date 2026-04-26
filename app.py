from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.exposure_action_agent import PERSONAS, generate_exposure_actions
from src.agents.grid_signal_agent import detect_grid_signals
from src.agents.price_impact_agent import detect_price_signals
from src.agents.reasoning_agent import reason_about_market
from src.data.cache import load_last_success
from src.data.ercot_client import SETTLEMENT_POINTS, ErcotClient, ErcotDataError, bundle_to_dict
from src.utils.plotting import load_chart, price_chart, price_load_overlay_chart, renewable_chart
from src.utils.schemas import DISCLAIMER
from src.utils.time_utils import briefing_filename

st.set_page_config(page_title="ERCOT Grid-to-Price Intelligence Agent", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Pulling ERCOT public data...")
def load_ercot_data(settlement_point: str, hours: int) -> dict:
    client = ErcotClient()
    return bundle_to_dict(client.fetch_bundle(settlement_point, hours))


def latest_value(df: pd.DataFrame, column: str):
    if df.empty or column not in df.columns:
        return None
    clean = df.dropna(subset=[column]).sort_values("timestamp")
    return None if clean.empty else clean.iloc[-1][column]


def trend(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns or len(df.dropna(subset=[column])) < 2:
        return "insufficient data"
    clean = df.dropna(subset=[column]).sort_values("timestamp")
    current = float(clean.iloc[-1][column])
    baseline = float(clean.iloc[:-1][column].tail(min(8, len(clean) - 1)).mean())
    if baseline == 0:
        return "insufficient data"
    change = (current - baseline) / abs(baseline) * 100
    return f"{change:+.1f}% vs rolling baseline"


def snapshot_field(label: str, value: str) -> None:
    st.markdown(f"**{label}**")
    st.write(value)


def money(value) -> str:
    return "insufficient data" if value is None else f"${float(value):,.2f}/MWh"


def percent(value) -> str:
    return "insufficient data" if value is None else f"{float(value):+,.1f}%"


def display_label(value) -> str:
    if value is None:
        return "Insufficient Data"
    text = str(value).strip()
    if not text:
        return "Insufficient Data"
    text = text.replace("real-time", "Real-Time").replace("day-ahead", "Day-Ahead")
    replacements = {
        "rt": "RT",
        "da": "DA",
        "ercot": "ERCOT",
    }
    words = []
    for word in text.replace("/", " / ").split():
        lower = word.lower()
        words.append(replacements.get(lower, word if any(char.isupper() for char in word) else word.capitalize()))
    return " ".join(words)


def display_sentence(value) -> str:
    if value is None:
        return "Insufficient data"
    text = str(value).strip()
    if not text:
        return "Insufficient data"
    return text[0].upper() + text[1:]


def signal_evidence_text(signal: dict) -> str:
    rows = signal.get("evidence", [])
    if not rows:
        return "No supporting evidence rows were available for this signal."
    parts = []
    for row in rows:
        metric = row.get("metric", "metric")
        observed = row.get("observed_value", "insufficient data")
        baseline = row.get("baseline", "insufficient data")
        timestamp = row.get("timestamp", "")
        parts.append(f"{metric}: observed {observed}, baseline {baseline}" + (f" at {timestamp}" if timestamp else ""))
    return "; ".join(parts)


def render_list(items: list[str], empty: str = "None identified") -> None:
    if not items:
        st.write(empty)
        return
    for item in items:
        st.write(f"- {display_sentence(item)}")


def evidence_table(*groups: list[dict]) -> pd.DataFrame:
    rows = []
    for group in groups:
        for item in group:
            rows.extend(item.get("evidence", []))
    if not rows:
        return pd.DataFrame(columns=["metric", "observed_value", "baseline", "timestamp", "source_dataset"])
    return pd.DataFrame(rows).drop_duplicates()


def markdown_briefing(settlement_point: str, hours: int, persona: str, grid_signals: list[dict], price_signals: list[dict], reasoning: dict, exposure: dict, evidence: pd.DataFrame) -> str:
    lines = [
        f"# ERCOT Grid-to-Price Intelligence Brief",
        "",
        f"- Settlement point: `{settlement_point}`",
        f"- Window: last {hours} hours",
        f"- Persona: {persona}",
        "",
        "## Observed Grid Signals",
    ]
    for signal in grid_signals:
        lines.append(f"- {display_label(signal.get('signal_name'))}: {display_label(signal.get('direction'))} ({signal.get('confidence')})")
    lines += ["", "## Observed Price Signals"]
    for signal in price_signals:
        lines.append(f"- {display_label(signal.get('price_signal'))}: current ${signal.get('current_price')}/MWh, baseline ${signal.get('baseline_price')}/MWh ({signal.get('confidence')})")
    lines += [
        "",
        "## Interpretation",
        display_sentence(reasoning.get("market_read") or reasoning.get("explanation", "insufficient data")),
        f"Likely drivers: {', '.join(display_sentence(item) for item in reasoning.get('likely_drivers', [])) or display_sentence(reasoning.get('likely_driver', 'insufficient data'))}",
        f"Confidence: {reasoning.get('confidence', 'Low')}",
        "",
        "## Supporting Observations",
    ]
    lines.extend([f"- {display_sentence(item)}" for item in reasoning.get("supporting_observations", [])] or ["- Insufficient data"])
    lines += ["", "## Unmodeled Factors"]
    lines.extend([f"- {display_sentence(item)}" for item in reasoning.get("unmodeled_factors", [])] or ["- Insufficient data"])
    lines += [
        "",
        "## Exposure and Possible Actions",
        f"Exposure: {display_sentence(exposure.get('exposure'))}",
    ]
    lines.extend([f"- {display_sentence(action)}" for action in exposure.get("possible_actions", [])])
    lines += ["", "## Linked Signals"]
    lines.extend([f"- {display_label(signal)}" for signal in exposure.get("linked_signals", [])] or ["- Insufficient data"])
    lines += ["", "## Evidence"]
    if evidence.empty:
        lines.append("insufficient data")
    else:
        lines.append(evidence.to_markdown(index=False))
    lines += ["", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)


def save_briefing(content: str) -> Path:
    out_dir = Path("outputs/briefings")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / briefing_filename()
    path.write_text(content)
    return path


st.title("ERCOT Grid-to-Price Intelligence Agent")

with st.sidebar:
    settlement_point = st.selectbox("Settlement point", SETTLEMENT_POINTS, index=0)
    hours = st.selectbox("Time window", [6, 12, 24, 48], index=2, format_func=lambda h: f"{h}h")
    persona = st.selectbox("Persona", PERSONAS, index=0)
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
    last = load_last_success()
    st.caption(f"Last successful pull: {last.get('last_refresh') if last else 'None yet'}")

try:
    data = load_ercot_data(settlement_point, hours)
except ErcotDataError as exc:
    st.error(str(exc))
    st.stop()

rt_df = data["rt_prices"]
da_df = data["da_prices"]
load_df = data["load"]
wind_df = data["wind"]
solar_df = data["solar"]

grid_signals = detect_grid_signals(load_df, wind_df, solar_df)
price_signals = detect_price_signals(rt_df, da_df)
reasoning = reason_about_market(grid_signals, price_signals)
exposure = generate_exposure_actions(persona, grid_signals, price_signals, reasoning)
evidence = evidence_table(grid_signals, price_signals, [reasoning])

st.subheader("Market Snapshot")
current_rt = latest_value(rt_df, "price")
current_da = latest_value(da_df, "price")
spread = current_rt - current_da if current_rt is not None and current_da is not None else None
price_cols = st.columns(3)
price_cols[0].metric("Current RT price", "insufficient data" if current_rt is None else f"${current_rt:,.2f}/MWh")
price_cols[1].metric("Day-ahead price", "insufficient data" if current_da is None else f"${current_da:,.2f}/MWh")
price_cols[2].metric("RT vs DA spread", "insufficient data" if spread is None else f"${spread:,.2f}/MWh")
trend_cols = st.columns(3)
with trend_cols[0]:
    snapshot_field("Load trend", trend(load_df, "load_mw"))
with trend_cols[1]:
    snapshot_field("Wind trend", trend(wind_df, "wind_mw"))
with trend_cols[2]:
    snapshot_field("Solar trend", trend(solar_df, "solar_mw"))

st.subheader("Charts")
st.plotly_chart(price_load_overlay_chart(rt_df, da_df, load_df), use_container_width=True)
st.plotly_chart(price_chart(rt_df, da_df), use_container_width=True)
left, right = st.columns(2)
left.plotly_chart(load_chart(load_df), use_container_width=True)
right.plotly_chart(renewable_chart(wind_df, solar_df), use_container_width=True)

st.subheader("Agent Output")
st.markdown(f"**Market Read:** {display_sentence(reasoning.get('market_read') or reasoning.get('explanation', 'insufficient data'))}")
summary_cols = st.columns([1, 2])
summary_cols[0].metric("Confidence", reasoning.get("confidence", "Low"))
with summary_cols[1]:
    st.markdown("**Likely Drivers**")
    render_list(reasoning.get("likely_drivers", []) or [reasoning.get("likely_driver", "insufficient data")])

tab_reasoning, tab_grid, tab_price, tab_exposure, tab_raw = st.tabs(["Reasoning", "Grid Signals", "Price Signals", "Exposure", "Raw Data"])

with tab_reasoning:
    st.markdown("**Supporting Observations**")
    render_list(reasoning.get("supporting_observations", []), "No supporting observations were identified.")
    st.markdown("**Unmodeled Factors To Check**")
    render_list(reasoning.get("unmodeled_factors", []), "No unmodeled factors were identified.")
    st.markdown("**Caveats**")
    render_list(reasoning.get("caveats", []))

with tab_grid:
    for signal in grid_signals:
        st.markdown(f"**{display_label(signal.get('signal_name', 'Grid signal'))}**")
        st.write(f"Direction: {display_label(signal.get('direction', 'insufficient data'))}")
        if signal.get("magnitude") is not None:
            st.write(f"Magnitude: {percent(signal.get('magnitude'))}")
        st.write(f"Confidence: {signal.get('confidence', 'Low')}")
        st.caption(signal_evidence_text(signal))

with tab_price:
    for signal in price_signals:
        st.markdown(f"**{display_label(signal.get('price_signal', 'Price signal'))}**")
        st.write(f"Current Real-Time Price: {money(signal.get('current_price'))}")
        st.write(f"Rolling Baseline Price: {money(signal.get('baseline_price'))}")
        st.write(f"Day-Ahead Price: {money(signal.get('day_ahead_price'))}")
        st.write(f"RT vs DA Spread: {money(signal.get('rt_da_spread'))}")
        if signal.get("price_change") is not None:
            st.write(f"Price Change: {percent(signal.get('price_change'))}")
        st.write(f"Confidence: {signal.get('confidence', 'Low')}")
        st.caption(signal_evidence_text(signal))

with tab_exposure:
    st.markdown(f"**Exposure:** {display_sentence(exposure.get('exposure'))}")
    st.markdown("**Data-Grounded Checks**")
    render_list(exposure.get("possible_actions", []))
    st.markdown("**Linked Signals**")
    render_list([display_label(signal) for signal in exposure.get("linked_signals", [])], "No material linked signals.")
    st.caption(DISCLAIMER)

with tab_raw:
    with st.expander("Raw grid signals"):
        st.json(grid_signals)
    with st.expander("Raw price signals"):
        st.json(price_signals)
    with st.expander("Raw reasoning output"):
        st.json(reasoning)
    with st.expander("Raw exposure output"):
        st.json(exposure)

st.subheader("Evidence Table")
st.dataframe(evidence, use_container_width=True)

st.subheader("Export")
brief = markdown_briefing(settlement_point, hours, persona, grid_signals, price_signals, reasoning, exposure, evidence)
saved_path = save_briefing(brief)
st.download_button("Export briefing as markdown", brief, file_name=saved_path.name, mime="text/markdown")
st.caption(f"Saved latest briefing to `{saved_path}`")
