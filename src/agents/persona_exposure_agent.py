from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.schemas import CONFIDENCE_LOW, CONFIDENCE_MEDIUM
from src.utils.time_utils import iso_ts


PERSONA_CARDS = [
    "Data center operator",
    "Industrial energy buyer",
    "Power trader",
    "Retail electricity provider",
    "Renewable operator",
    "Battery/storage operator",
]

RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3}


@dataclass
class MetricSnapshot:
    value: float | None
    timestamp: str


def _latest_numeric(df: pd.DataFrame, value_col: str) -> MetricSnapshot:
    if df.empty or value_col not in df.columns or "timestamp" not in df.columns:
        return MetricSnapshot(None, "")
    clean = df.dropna(subset=[value_col]).sort_values("timestamp")
    if clean.empty:
        return MetricSnapshot(None, "")
    latest = clean.iloc[-1]
    return MetricSnapshot(float(latest[value_col]), iso_ts(latest["timestamp"]))


def _rolling_baseline(df: pd.DataFrame, value_col: str, lookback: int = 8) -> float | None:
    if df.empty or value_col not in df.columns:
        return None
    clean = df.dropna(subset=[value_col]).sort_values("timestamp")
    if len(clean) < 2:
        return None
    baseline = clean.iloc[:-1][value_col].tail(min(lookback, len(clean) - 1)).mean()
    return None if pd.isna(baseline) else float(baseline)


def _pct_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in (None, 0):
        return None
    return (current - baseline) / abs(baseline) * 100


def _signal_names(signals: list[dict], key: str) -> set[str]:
    return {str(signal.get(key, "")) for signal in signals}


def _risk_label(score: int) -> str:
    if score >= 5:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def _confidence(data_points: list[Any], has_signal: bool) -> str:
    available = sum(item is not None for item in data_points)
    if available >= 4 and has_signal:
        return "High"
    if available >= 2:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def build_market_pulse(rt_df: pd.DataFrame, da_df: pd.DataFrame, load_df: pd.DataFrame, wind_df: pd.DataFrame, solar_df: pd.DataFrame, grid_signals: list[dict], price_signals: list[dict]) -> dict:
    rt = _latest_numeric(rt_df, "price")
    da = _latest_numeric(da_df, "price")
    load = _latest_numeric(load_df, "load_mw")
    wind = _latest_numeric(wind_df, "wind_mw")
    solar = _latest_numeric(solar_df, "solar_mw")

    spread = rt.value - da.value if rt.value is not None and da.value is not None else None
    load_change = _pct_change(load.value, _rolling_baseline(load_df, "load_mw"))
    wind_change = _pct_change(wind.value, _rolling_baseline(wind_df, "wind_mw"))
    solar_change = _pct_change(solar.value, _rolling_baseline(solar_df, "solar_mw"))
    renewable_total = None if wind.value is None and solar.value is None else (wind.value or 0) + (solar.value or 0)
    renewable_share = renewable_total / load.value * 100 if renewable_total is not None and load.value not in (None, 0) else None

    price_names = _signal_names(price_signals, "price_signal")
    grid_names = _signal_names(grid_signals, "signal_name")

    # Market risk is intentionally transparent: price stress matters most, then
    # physical tightening and RT/DA divergence add context.
    risk_score = 0
    if rt.value is not None:
        if rt.value >= 150:
            risk_score += 4
        elif rt.value >= 75:
            risk_score += 2
        elif rt.value < 0:
            risk_score += 1
    if spread is not None:
        if spread >= 75:
            risk_score += 3
        elif spread >= 25:
            risk_score += 2
        elif spread <= -25:
            risk_score += 1
    if "real-time price volatility" in price_names:
        risk_score += 2
    if {"real-time price spike", "real-time premium to day-ahead"} & price_names:
        risk_score += 2
    if "possible supply tightening" in grid_names:
        risk_score += 2
    if "demand spike" in grid_names:
        risk_score += 1
    if {"wind generation drop", "solar generation drop"} & grid_names:
        risk_score += 1

    return {
        "latest_rt_price": rt.value,
        "latest_rt_timestamp": rt.timestamp,
        "latest_da_price": da.value,
        "latest_da_timestamp": da.timestamp,
        "rt_da_spread": spread,
        "latest_load_mw": load.value,
        "latest_load_timestamp": load.timestamp,
        "latest_wind_mw": wind.value,
        "latest_solar_mw": solar.value,
        "renewable_total_mw": renewable_total,
        "renewable_share": renewable_share,
        "load_change_pct": load_change,
        "wind_change_pct": wind_change,
        "solar_change_pct": solar_change,
        "risk_score": risk_score,
        "risk_label": _risk_label(risk_score),
        "confidence": _confidence([rt.value, da.value, load.value, wind.value, solar.value], bool(price_names or grid_names)),
    }


def _raise_level(level: str, risk_label: str) -> str:
    return risk_label if RISK_ORDER[risk_label] > RISK_ORDER[level] else level


def _drivers(reasoning: dict, pulse: dict) -> list[str]:
    drivers = [item for item in reasoning.get("likely_drivers", []) if item and item != "insufficient data"]
    if not drivers and reasoning.get("likely_driver") and reasoning.get("likely_driver") != "insufficient data":
        drivers = [reasoning["likely_driver"]]
    if pulse.get("rt_da_spread") is not None:
        drivers.append(f"RT minus DA spread is {pulse['rt_da_spread']:+.2f}/MWh")
    return drivers or ["No material aligned signal in the selected window"]


def build_persona_exposures(pulse: dict, grid_signals: list[dict], price_signals: list[dict], reasoning: dict) -> list[dict]:
    price_names = _signal_names(price_signals, "price_signal")
    grid_names = _signal_names(grid_signals, "signal_name")
    risk = pulse.get("risk_label", "Low")
    spread = pulse.get("rt_da_spread")
    rt_price = pulse.get("latest_rt_price")
    loose = spread is not None and spread < -25
    premium = spread is not None and spread > 25
    high_price = rt_price is not None and rt_price >= 75
    negative_price = rt_price is not None and rt_price < 0
    tight = "possible supply tightening" in grid_names or "demand spike" in grid_names
    renewable_down = bool({"wind generation drop", "solar generation drop"} & grid_names)
    renewable_up = "renewable generation recovery" in grid_names
    volatile = "real-time price volatility" in price_names
    drivers = _drivers(reasoning, pulse)
    assumptions = [
        "Exposure assumes the selected hub is relevant to the user's settlement or hedge position.",
        "Contract pass-through terms, nodal congestion, outages, and ancillary charges are not fully modeled.",
    ]

    # Persona scoring uses the same market pulse, then adjusts by how each user
    # type is economically exposed to high prices, negative prices, spreads, or volatility.
    cards = []
    specs = [
        (
            "Data center operator",
            "Medium" if high_price or premium or tight else "Low",
            "Real-time-indexed load and pass-through charges can move quickly when prices or scarcity risk rise.",
            "RT prices, RT/DA spread, load ramps, reserve scarcity, and contract pass-through language.",
            "Review curtailment options, load-shift windows, and hedge coverage for the next intervals.",
        ),
        (
            "Industrial energy buyer",
            "Medium" if high_price or premium or tight else "Low",
            "Large load exposure is most sensitive to elevated real-time prices and imbalance charges.",
            "Whether the premium persists, whether load keeps rising, and whether renewable output recovers.",
            "Compare current interval exposure against fixed-price, DA, or hedged coverage.",
        ),
        (
            "Power trader",
            "High" if premium or loose or volatile else "Medium" if high_price or negative_price else "Low",
            "RT/DA divergence and volatility create the clearest trading-relevant signal in this MVP.",
            "Spread persistence, volatility, congestion reports, outages, and reserve conditions.",
            "Validate the signal against constraints and liquidity before acting on the spread.",
        ),
        (
            "Retail electricity provider",
            "High" if high_price and tight else "Medium" if high_price or premium or volatile else "Low",
            "Portfolio margin risk rises when real-time supply costs diverge from customer pricing assumptions.",
            "Load forecast error, customer demand, RT/DA spread, and hedge coverage.",
            "Check hedge ratios, pass-through mechanisms, and short-term procurement needs.",
        ),
        (
            "Renewable operator",
            "High" if negative_price else "Medium" if renewable_down or loose else "Low",
            "Revenue risk depends on renewable output, curtailment exposure, and negative or discounted pricing.",
            "Negative prices, curtailment risk, renewable output versus baseline, and local congestion.",
            "Review dispatch, curtailment, and hedge/PPA settlement terms for affected intervals.",
        ),
        (
            "Battery/storage operator",
            "High" if premium or volatile or negative_price else "Medium" if high_price or loose else "Low",
            "Storage value is sensitive to RT volatility, charging discounts, and discharge premiums.",
            "Charge opportunities during discounts, discharge value during spikes, and operational constraints.",
            "Evaluate charge/discharge timing against state of charge, offers, and expected next-interval spreads.",
        ),
    ]

    for persona, base_level, driver_note, watch, action in specs:
        level = _raise_level(base_level, risk) if persona in {"Data center operator", "Power trader", "Retail electricity provider", "Battery/storage operator"} else base_level
        if loose and persona in {"Data center operator", "Industrial energy buyer"}:
            action = "Assess whether discounted real-time exposure creates a short-term operating or procurement advantage."
        if renewable_up and persona == "Renewable operator":
            driver_note = "Renewable output recovery can lift production but may pressure prices if supply is abundant."
        cards.append(
            {
                "persona": persona,
                "exposure_level": level,
                "main_drivers": drivers + [driver_note],
                "what_to_watch_next": watch,
                "possible_action": action,
                "confidence": _confidence(
                    [
                        pulse.get("latest_rt_price"),
                        pulse.get("latest_da_price"),
                        pulse.get("latest_load_mw"),
                        pulse.get("latest_wind_mw"),
                        pulse.get("latest_solar_mw"),
                    ],
                    level != "Low",
                ),
                "assumptions": assumptions,
            },
        )
    return cards


def build_agent_brief(pulse: dict, persona_cards: list[dict], reasoning: dict) -> dict:
    most_exposed = [card["persona"] for card in persona_cards if card["exposure_level"] == "High"]
    if not most_exposed:
        most_exposed = [card["persona"] for card in persona_cards if card["exposure_level"] == "Medium"][:3]

    changes = []
    for label, key in [("load", "load_change_pct"), ("wind", "wind_change_pct"), ("solar", "solar_change_pct")]:
        value = pulse.get(key)
        if value is not None:
            changes.append(f"{label} is {value:+.1f}% versus its recent rolling baseline")
    if pulse.get("rt_da_spread") is not None:
        changes.append(f"real-time is {pulse['rt_da_spread']:+.2f}/MWh versus day-ahead")

    return {
        "current_market_risk": pulse.get("risk_label", "Low"),
        "what_changed_recently": changes or ["Insufficient recent data to calculate changes versus baseline"],
        "main_drivers": reasoning.get("likely_drivers") or [reasoning.get("likely_driver", "No material aligned signal")],
        "most_exposed_personas": most_exposed or ["No persona shows elevated exposure in the selected window"],
        "next_6_hour_watch_window": [
            "Track whether RT prices keep diverging from day-ahead.",
            "Watch load forecast revisions and renewable output versus rolling baseline.",
            "Check congestion, outage, reserve, and ancillary service reports before attributing causality.",
        ],
        "recommended_actions": [
            "Compare current exposure against DA, fixed-price, or hedged coverage.",
            "Refresh after the next ERCOT interval and confirm whether the signal persists.",
            "Escalate only if price, spread, and physical grid signals remain aligned.",
        ],
        "assumptions": [
            "This is rule-based market intelligence, not trading or financial advice.",
            "The selected settlement point is treated as the relevant price reference.",
            "Unmodeled congestion, outages, ancillary scarcity, and nodal effects may dominate system-wide signals.",
        ],
        "confidence": pulse.get("confidence", CONFIDENCE_LOW),
    }
