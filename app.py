from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.grid_signal_agent import detect_grid_signals
from src.agents.persona_exposure_agent import build_agent_brief, build_market_pulse, build_persona_exposures
from src.agents.price_impact_agent import detect_price_signals
from src.agents.reasoning_agent import reason_about_market
from src.data.cache import load_last_success
from src.data.ercot_client import SETTLEMENT_POINTS, ErcotClient, ErcotDataError, bundle_to_dict
from src.utils.plotting import load_chart, price_chart, renewable_chart
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


def mw(value) -> str:
    return "insufficient data" if value is None else f"{float(value):,.0f} MW"


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


def risk_help(risk: str) -> str:
    descriptions = {
        "Low": "Low: prices and physical grid signals are not showing material stress in this window.",
        "Medium": "Medium: one or more price, spread, volatility, or grid tightening signals is elevated.",
        "High": "High: price stress, RT/DA divergence, volatility, or supply tightening is materially elevated.",
    }
    return descriptions.get(risk, "Risk is based on deterministic price and grid heuristics.")


def render_card_list(items: list[str], empty: str = "Insufficient data") -> None:
    if not items:
        st.write(empty)
        return
    for item in items:
        st.write(f"- {display_sentence(item)}")


def first_items(items: list[str], count: int = 2) -> list[str]:
    return items[:count] if items else []


def analyst_lede(agent_brief: dict, pulse: dict, settlement_point: str, hours: int) -> str:
    risk = agent_brief.get("current_market_risk", "Low")
    spread = pulse.get("rt_da_spread")
    spread_text = "without a confirmed RT/DA spread" if spread is None else f"with RT trading {spread:+.2f}/MWh versus day-ahead"
    exposed = ", ".join(agent_brief.get("most_exposed_personas", [])) or "no persona showing elevated exposure"
    return (
        f"Market risk is **{risk}** at `{settlement_point}` over the last {hours} hours, "
        f"{spread_text}. The most exposed segment is: {exposed}."
    )


def markdown_briefing(settlement_point: str, hours: int, market_pulse: dict, persona_cards: list[dict], agent_brief: dict, grid_signals: list[dict], price_signals: list[dict], reasoning: dict, evidence: pd.DataFrame) -> str:
    lines = [
        f"# ERCOT Grid-to-Price Intelligence Brief",
        "",
        f"- Settlement point: `{settlement_point}`",
        f"- Window: last {hours} hours",
        f"- Current market risk: {agent_brief.get('current_market_risk', 'Low')}",
        f"- Confidence: {agent_brief.get('confidence', 'Low')}",
        "",
        "## Market Pulse",
        f"- Latest RT price: {money(market_pulse.get('latest_rt_price'))}",
        f"- Latest DA price: {money(market_pulse.get('latest_da_price'))}",
        f"- RT minus DA spread: {money(market_pulse.get('rt_da_spread'))}",
        f"- Latest load forecast: {mw(market_pulse.get('latest_load_mw'))}",
        f"- Wind generation: {mw(market_pulse.get('latest_wind_mw'))}",
        f"- Solar generation: {mw(market_pulse.get('latest_solar_mw'))}",
        "",
        "## Agent Brief",
        f"- Most exposed personas: {', '.join(agent_brief.get('most_exposed_personas', []))}",
        "",
        "### What Changed Recently",
    ]
    lines.extend([f"- {display_sentence(item)}" for item in agent_brief.get("what_changed_recently", [])] or ["- Insufficient data"])
    lines += ["", "### Main Drivers"]
    lines.extend([f"- {display_sentence(item)}" for item in agent_brief.get("main_drivers", [])] or ["- Insufficient data"])
    lines += ["", "### Recommended Actions"]
    lines.extend([f"- {display_sentence(item)}" for item in agent_brief.get("recommended_actions", [])] or ["- Insufficient data"])
    lines += ["", "## Persona Exposure"]
    for card in persona_cards:
        lines.append(f"### {card['persona']}")
        lines.append(f"- Exposure level: {card['exposure_level']}")
        lines.append(f"- Confidence: {card['confidence']}")
        lines.append(f"- What to watch next: {card['what_to_watch_next']}")
        lines.append(f"- Possible action: {card['possible_action']}")
    lines += [
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
    ]
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
st.info("Uses public ERCOT data only. This is market intelligence, not trading advice. Human review required.")

with st.sidebar:
    settlement_point = st.selectbox("Settlement point", SETTLEMENT_POINTS, index=0)
    hours = st.selectbox("Time window", [6, 12, 24, 48], index=2, format_func=lambda h: f"{h}h")
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
market_pulse = build_market_pulse(rt_df, da_df, load_df, wind_df, solar_df, grid_signals, price_signals)
persona_cards = build_persona_exposures(market_pulse, grid_signals, price_signals, reasoning)
agent_brief = build_agent_brief(market_pulse, persona_cards, reasoning)
evidence = evidence_table(grid_signals, price_signals, [reasoning])

tab_pulse, tab_persona, tab_brief, tab_charts, tab_signals, tab_raw = st.tabs(
    ["Market Pulse", "Persona Exposure", "Agent Brief", "Charts", "Signals & Evidence", "Raw Data"],
)

with tab_pulse:
    st.subheader("Market Pulse")
    st.caption("One-screen readout of price stress, grid conditions, and the current rule-based market risk label.")
    st.markdown(analyst_lede(agent_brief, market_pulse, settlement_point, hours))
    pulse_cols = st.columns(4)
    pulse_cols[0].metric("Market risk", market_pulse["risk_label"], help=risk_help(market_pulse["risk_label"]))
    pulse_cols[1].metric("Latest RT price", money(market_pulse.get("latest_rt_price")))
    pulse_cols[2].metric("Latest DA price", money(market_pulse.get("latest_da_price")))
    pulse_cols[3].metric("RT minus DA", money(market_pulse.get("rt_da_spread")))
    grid_cols = st.columns(4)
    grid_cols[0].metric("Latest load forecast", mw(market_pulse.get("latest_load_mw")), percent(market_pulse.get("load_change_pct")) if market_pulse.get("load_change_pct") is not None else None)
    grid_cols[1].metric("Wind generation", mw(market_pulse.get("latest_wind_mw")), percent(market_pulse.get("wind_change_pct")) if market_pulse.get("wind_change_pct") is not None else None)
    grid_cols[2].metric("Solar generation", mw(market_pulse.get("latest_solar_mw")), percent(market_pulse.get("solar_change_pct")) if market_pulse.get("solar_change_pct") is not None else None)
    grid_cols[3].metric("Renewable share of load", "insufficient data" if market_pulse.get("renewable_share") is None else f"{market_pulse['renewable_share']:,.1f}%")
    with st.container(border=True):
        st.markdown("**Current market read**")
        st.write(display_sentence(reasoning.get("market_read") or reasoning.get("explanation", "insufficient data")))
    with st.expander("Latest data timestamps"):
        st.write(f"RT: {market_pulse.get('latest_rt_timestamp') or 'insufficient data'}")
        st.write(f"DA: {market_pulse.get('latest_da_timestamp') or 'insufficient data'}")
        st.write(f"Load: {market_pulse.get('latest_load_timestamp') or 'insufficient data'}")

with tab_persona:
    st.subheader("Persona Exposure")
    st.caption("Compact exposure map by user type. Cards use the same deterministic market pulse, then adjust for each persona's economic sensitivity.")
    for row_start in range(0, len(persona_cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, persona_cards[row_start : row_start + 3]):
            with col.container(border=True):
                st.markdown(f"**{card['persona']}**")
                metric_cols = st.columns(2)
                metric_cols[0].metric("Exposure", card["exposure_level"])
                metric_cols[1].metric("Confidence", card["confidence"])
                st.markdown("**Drivers**")
                render_card_list(first_items(card["main_drivers"], 2))
                st.markdown("**Watch**")
                st.caption(display_sentence(card["what_to_watch_next"]))
                st.markdown("**Action**")
                st.caption(display_sentence(card["possible_action"]))
                with st.expander("Assumptions and full drivers"):
                    render_card_list(card["main_drivers"])
                    render_card_list(card["assumptions"])

with tab_brief:
    st.subheader("Agent Brief")
    st.caption("Executive-style market note generated from the current grid and price signals.")
    st.markdown(analyst_lede(agent_brief, market_pulse, settlement_point, hours))
    brief_cols = st.columns([1, 1, 2])
    brief_cols[0].metric("Risk", agent_brief["current_market_risk"])
    brief_cols[1].metric("Confidence", agent_brief["confidence"])
    with brief_cols[2]:
        st.markdown("**Most exposed**")
        st.write(", ".join(agent_brief["most_exposed_personas"]))
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**What changed recently**")
        render_card_list(agent_brief["what_changed_recently"])
        st.markdown("**Main drivers**")
        render_card_list(agent_brief["main_drivers"])
    with right.container(border=True):
        st.markdown("**Next 6-hour watch window**")
        render_card_list(agent_brief["next_6_hour_watch_window"])
        st.markdown("**Recommended actions**")
        render_card_list(agent_brief["recommended_actions"])
    with st.expander("Assumptions and confidence basis"):
        render_card_list(agent_brief["assumptions"])
    st.caption("Brief is generated from deterministic structured logic. LLM polishing is not enabled in this build.")

with tab_charts:
    st.subheader("Charts")
    st.caption("Detailed price and grid context for users who want to inspect the data behind the pulse and brief.")
    st.plotly_chart(price_chart(rt_df, da_df), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(load_chart(load_df), use_container_width=True)
    right.plotly_chart(renewable_chart(wind_df, solar_df), use_container_width=True)

with tab_signals:
    st.subheader("Signal Detail")
    st.caption("Explainability layer showing the rule-based grid and price signals used by the exposure cards.")
    st.markdown("**Grid Signals**")
    for signal in grid_signals:
        with st.container(border=True):
            st.markdown(f"**{display_label(signal.get('signal_name', 'Grid signal'))}**")
            st.write(f"Direction: {display_label(signal.get('direction', 'insufficient data'))}")
            if signal.get("magnitude") is not None:
                st.write(f"Magnitude: {percent(signal.get('magnitude'))}")
            st.write(f"Confidence: {signal.get('confidence', 'Low')}")
            st.caption(signal_evidence_text(signal))
    st.markdown("**Price Signals**")
    for signal in price_signals:
        with st.container(border=True):
            st.markdown(f"**{display_label(signal.get('price_signal', 'Price signal'))}**")
            st.write(f"Current Real-Time Price: {money(signal.get('current_price'))}")
            st.write(f"Rolling Baseline Price: {money(signal.get('baseline_price'))}")
            st.write(f"Day-Ahead Price: {money(signal.get('day_ahead_price'))}")
            st.write(f"RT vs DA Spread: {money(signal.get('rt_da_spread'))}")
            if signal.get("price_change") is not None:
                st.write(f"Price Change: {percent(signal.get('price_change'))}")
            st.write(f"Confidence: {signal.get('confidence', 'Low')}")
            st.caption(signal_evidence_text(signal))
    st.markdown("**Evidence Table**")
    st.dataframe(evidence, use_container_width=True)

with tab_raw:
    st.caption("Raw structured outputs for debugging, validation, and demo transparency.")
    with st.expander("Raw market pulse"):
        st.json(market_pulse)
    with st.expander("Raw persona exposure cards"):
        st.json(persona_cards)
    with st.expander("Raw agent brief"):
        st.json(agent_brief)
    with st.expander("Raw grid signals"):
        st.json(grid_signals)
    with st.expander("Raw price signals"):
        st.json(price_signals)
    with st.expander("Raw reasoning output"):
        st.json(reasoning)

st.subheader("Export")
brief = markdown_briefing(settlement_point, hours, market_pulse, persona_cards, agent_brief, grid_signals, price_signals, reasoning, evidence)
saved_path = save_briefing(brief)
st.download_button("Export briefing as markdown", brief, file_name=saved_path.name, mime="text/markdown")
st.caption(f"Saved latest briefing to `{saved_path}`")
