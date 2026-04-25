from __future__ import annotations

from src.utils.schemas import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, ReasoningOutput


def _names(items: list[dict], key: str) -> list[str]:
    return [item.get(key, "") for item in items]


def _evidence(grid_signals: list[dict], price_signals: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in grid_signals + price_signals:
        out.extend(item.get("evidence", []))
    return out


def reason_about_market(grid_signals: list[dict], price_signals: list[dict]) -> dict:
    grid_names = _names(grid_signals, "signal_name")
    price_names = _names(price_signals, "price_signal")
    evidence = _evidence(grid_signals, price_signals)

    has_load_up = "demand spike" in grid_names
    has_wind_drop = "wind generation drop" in grid_names
    has_solar_drop = "solar generation drop" in grid_names
    has_tightening = "possible supply tightening" in grid_names
    has_renewable_recovery = "renewable generation recovery" in grid_names
    has_price_spike = "real-time price spike" in price_names or "real-time premium to day-ahead" in price_names
    has_price_drop = "real-time price drop" in price_names or "negative price signal" in price_names
    has_da_spread = "real-time premium to day-ahead" in price_names

    caveats = [
        "ERCOT grid and price data are observational; this workflow does not prove causality.",
        "Transmission constraints, outages, ancillary conditions, or local congestion may affect prices but are not fully modeled in v1.",
    ]

    if (has_tightening or has_load_up or has_wind_drop or has_solar_drop) and has_price_spike:
        drivers = []
        if has_load_up:
            drivers.append("higher demand")
        if has_wind_drop or has_solar_drop:
            drivers.append("lower renewable output")
        explanation = f"Grid conditions and real-time pricing moved in the same direction. The evidence points to {' and '.join(drivers) or 'physical grid tightening'} coinciding with upward real-time price pressure."
        return ReasoningOutput(
            explanation=explanation,
            likely_driver="supply tightening / demand-driven price pressure",
            confidence=CONFIDENCE_HIGH,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_renewable_recovery and has_price_drop:
        return ReasoningOutput(
            explanation="Renewable output recovered while real-time prices weakened or turned negative, which is consistent with surplus renewable pressure during the observed interval.",
            likely_driver="renewable recovery / surplus energy pressure",
            confidence=CONFIDENCE_HIGH,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_da_spread:
        return ReasoningOutput(
            explanation="Real-time prices diverged materially from day-ahead pricing. This suggests a real-time imbalance relative to the day-ahead expectation, but the available v1 grid signals do not fully explain the spread.",
            likely_driver="real-time imbalance versus day-ahead schedule",
            confidence=CONFIDENCE_MEDIUM,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_price_spike:
        return ReasoningOutput(
            explanation="Real-time prices increased without a strongly aligned load or renewable signal. A constraint, outage, reserve condition, or localized congestion could be involved, but confidence is low in v1.",
            likely_driver="possible constraint or outage",
            confidence=CONFIDENCE_LOW,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if not evidence:
        return ReasoningOutput(
            explanation="insufficient data",
            likely_driver="insufficient data",
            confidence=CONFIDENCE_LOW,
            evidence=[],
            caveats=["Required ERCOT datasets were missing or too sparse for rule-based interpretation."],
        ).__dict__

    return ReasoningOutput(
        explanation="Observed grid and price movements do not show a strong aligned signal in the selected window.",
        likely_driver="no material aligned signal",
        confidence=CONFIDENCE_LOW,
        evidence=evidence,
        caveats=caveats,
    ).__dict__
