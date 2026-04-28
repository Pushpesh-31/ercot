from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.grid_signal_agent import detect_grid_signals
from src.agents.location_intelligence_agent import (
    build_location_intelligence,
    price_spike_leaderboards,
    spread_leaderboard,
    top_location_explanations,
    volatility_leaderboard,
)
from src.agents.persona_exposure_agent import build_agent_brief, build_market_pulse, build_persona_exposures
from src.agents.price_impact_agent import detect_price_signals
from src.agents.reasoning_agent import reason_about_market
from src.data.cache import load_last_success
from src.data.ercot_client import SETTLEMENT_POINTS, ErcotClient, ErcotDataError, bundle_to_dict
from src.utils.plotting import load_chart, price_chart, renewable_chart
from src.utils.schemas import DISCLAIMER
from src.utils.time_utils import ERCOT_TZ, briefing_filename, now_ercot

st.set_page_config(page_title="ERCOT Grid-to-Price Intelligence Agent", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Pulling ERCOT public data...")
def load_ercot_data(settlement_point: str, hours: int) -> dict:
    client = ErcotClient()
    return bundle_to_dict(client.fetch_bundle(settlement_point, hours))


@st.cache_data(ttl=3600, show_spinner="Pulling ERCOT settlement point prices...")
def load_location_intelligence(hours: int) -> pd.DataFrame:
    try:
        client = ErcotClient()
        if hasattr(client, "fetch_settlement_point_prices"):
            prices = client.fetch_settlement_point_prices(hours)
        else:
            prices = fetch_settlement_point_prices_compat(client, hours)
        mapping = fetch_settlement_point_mapping_compat(client)
    except ErcotDataError:
        raise
    except Exception as exc:
        raise ErcotDataError(f"Location intelligence unavailable. Unable to pull ERCOT settlement point prices. Details: {exc}") from exc
    return build_location_intelligence(prices["rt_prices"], prices["da_prices"], mapping)


def fetch_settlement_point_prices_compat(client: ErcotClient, hours: int) -> dict[str, pd.DataFrame]:
    end = now_ercot()
    start = end - pd.Timedelta(hours=hours)
    fetch_start = start.floor("d")
    fetch_end = (end + pd.Timedelta(days=1)).floor("d")
    rt_raw = client.api.get_spp_real_time_15_min(fetch_start, end=fetch_end)
    da_raw = client.api.get_spp_day_ahead_hourly(fetch_start, end=fetch_end)
    return {
        "rt_prices": normalize_location_prices(rt_raw, start, end, "rt_price", "NP6-905-CD"),
        "da_prices": normalize_location_prices(da_raw, start, end, "da_price", "NP4-190-CD"),
    }


def fetch_settlement_point_mapping_compat(client: ErcotClient) -> pd.DataFrame:
    try:
        if hasattr(client, "fetch_settlement_point_mapping"):
            return client.fetch_settlement_point_mapping()
        from gridstatus import Ercot

        return Ercot().get_settlement_points_electrical_bus_mapping("latest")
    except Exception:
        return pd.DataFrame()


def normalize_location_prices(raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, price_name: str, source: str) -> pd.DataFrame:
    columns = ["timestamp", "settlement_point", price_name, "source_dataset"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    time_col = first_existing(raw, ["Interval Start", "Delivery Date", "deliveryDate"])
    loc_col = first_existing(raw, ["Location", "Settlement Point", "settlementPoint", "SettlementPoint"])
    price_col = first_existing(raw, ["Settlement Point Price", "SPP", "Price", "settlementPointPrice"])
    if not all([time_col, loc_col, price_col]):
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[time_col]),
            "settlement_point": raw[loc_col].astype(str),
            price_name: pd.to_numeric(raw[price_col], errors="coerce"),
            "source_dataset": source,
        },
    ).dropna(subset=["timestamp"])
    if out.empty:
        return out
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(ERCOT_TZ, ambiguous="NaT", nonexistent="NaT")
    return out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


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


def risk_badge(label: str) -> str:
    palette = {
        "High congestion": ("#991b1b", "#fee2e2"),
        "Medium congestion": ("#92400e", "#fef3c7"),
        "Low congestion": ("#166534", "#dcfce7"),
        "Extreme": ("#991b1b", "#fee2e2"),
        "Volatile": ("#92400e", "#fef3c7"),
        "Stable": ("#166534", "#dcfce7"),
    }
    color, background = palette.get(label, ("#374151", "#f3f4f6"))
    return (
        f"<span style='display:inline-block;padding:0.15rem 0.5rem;border-radius:999px;"
        f"font-size:0.8rem;font-weight:600;color:{color};background:{background};'>{label}</span>"
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
    location_hours = st.selectbox("Location intelligence window", [3, 6, 12], index=1, format_func=lambda h: f"{h}h")
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

try:
    location_df = load_location_intelligence(location_hours)
    location_error = None
except ErcotDataError as exc:
    location_df = pd.DataFrame()
    location_error = str(exc)

tab_pulse, tab_persona, tab_brief, tab_location, tab_charts, tab_signals, tab_raw = st.tabs(
    ["Market Pulse", "Persona Exposure", "Agent Brief", "Location Intelligence", "Charts", "Signals & Evidence", "Raw Data"],
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

with tab_location:
    st.subheader("Location Intelligence")
    st.caption("Top settlement-point anomalies from public ERCOT SPP data. Congestion labels are proxy signals based on price separation, spreads, and jumps.")
    if location_error:
        st.warning(f"Location intelligence unavailable: {location_error}")
    elif location_df.empty:
        st.info("Insufficient settlement point data to build location intelligence for the selected window.")
    else:
        latest_locations = location_df.sort_values("timestamp").groupby("settlement_point", as_index=False).tail(1)
        latest_timestamp = latest_locations["timestamp"].max()
        high_congestion = int((latest_locations["congestion_label"] == "High congestion").sum())
        medium_congestion = int((latest_locations["congestion_label"] == "Medium congestion").sum())
        extreme_volatility = int((latest_locations["volatility_label"] == "Extreme").sum())
        mapped_locations = int((latest_locations["mapping_confidence"] != "Unmapped fallback").sum()) if "mapping_confidence" in latest_locations.columns else 0
        loc_cols = st.columns(4)
        loc_cols[0].metric("Settlement points", f"{latest_locations['settlement_point'].nunique():,}")
        loc_cols[1].metric("Latest interval", "insufficient data" if pd.isna(latest_timestamp) else str(latest_timestamp))
        loc_cols[2].metric("High / medium signals", f"{high_congestion} / {medium_congestion}")
        loc_cols[3].metric("Mapped references", f"{mapped_locations:,}")

        high_prices, low_prices = price_spike_leaderboards(location_df)
        spreads = spread_leaderboard(location_df)
        volatility = volatility_leaderboard(location_df)
        top_anomalies = latest_locations.sort_values(
            ["congestion_score", "volatility_score", "deviation_from_reference"],
            ascending=[False, False, False],
        ).head(3)

        st.markdown("### Top congestion signals today")
        anomaly_cols = st.columns(3)
        for col, (_, row) in zip(anomaly_cols, top_anomalies.iterrows()):
            with col.container(border=True):
                st.markdown(f"**{row.get('settlement_point_display', row['settlement_point'])}**")
                st.markdown(risk_badge(row["congestion_label"]), unsafe_allow_html=True)
                st.metric("RT price", money(row.get("rt_price")))
                st.caption(f"Type: {display_label(row.get('settlement_point_type'))}")
                st.caption(f"Hub deviation: {money(row.get('deviation_from_reference_hub'))}")
                st.caption(f"Mapping: {row.get('mapping_confidence', 'Unmapped fallback')}")
                if pd.notna(row.get("rt_da_spread")):
                    st.caption(f"RT-DA spread: {money(row.get('rt_da_spread'))}")
                st.markdown(risk_badge(row["volatility_label"]), unsafe_allow_html=True)

        st.markdown("**Agent readout**")
        for explanation in top_location_explanations(location_df):
            st.write(f"- {explanation}")

        st.markdown("**Decision tables**")
        table_cols = st.columns(3)
        with table_cols[0]:
            st.markdown("**Highest RT prices**")
            st.dataframe(high_prices.head(5), use_container_width=True, hide_index=True)
        with table_cols[1]:
            st.markdown("**Widest RT-DA spreads**")
            if spreads.empty:
                st.info("DA prices unavailable.")
            else:
                st.dataframe(spreads.head(5), use_container_width=True, hide_index=True)
        with table_cols[2]:
            st.markdown("**Most volatile**")
            st.dataframe(volatility.head(5), use_container_width=True, hide_index=True)

        with st.expander("Full leaderboards"):
            price_tab, spread_tab, volatility_tab = st.tabs(["Price Spike Leaderboard", "Spread Leaderboard", "Volatility Leaderboard"])
            with price_tab:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Top 10 highest RT prices**")
                    st.dataframe(high_prices, use_container_width=True, hide_index=True)
                    if not high_prices.empty:
                        st.bar_chart(high_prices.set_index("settlement_point_display")["rt_price"])
                with right:
                    st.markdown("**Top 10 lowest RT prices**")
                    st.dataframe(low_prices, use_container_width=True, hide_index=True)
                    if not low_prices.empty:
                        st.bar_chart(low_prices.set_index("settlement_point_display")["rt_price"])
            with spread_tab:
                st.markdown("**Top 10 highest RT minus DA spreads**")
                if spreads.empty:
                    st.info("Day-ahead settlement point prices are unavailable for this window, so spread ranking is not available.")
                else:
                    st.dataframe(spreads, use_container_width=True, hide_index=True)
                    st.bar_chart(spreads.set_index("settlement_point_display")["rt_da_spread"])
            with volatility_tab:
                st.markdown("**Top 10 rolling volatility scores**")
                st.dataframe(volatility, use_container_width=True, hide_index=True)
                if not volatility.empty:
                    st.bar_chart(volatility.set_index("settlement_point_display")["volatility_score"])

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
    with st.expander("Raw location intelligence sample"):
        st.dataframe(location_df.head(100), use_container_width=True)
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
