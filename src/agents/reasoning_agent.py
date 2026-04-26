from __future__ import annotations

from src.utils.schemas import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, ReasoningOutput


def _names(items: list[dict], key: str) -> list[str]:
    return [item.get(key, "") for item in items]


def _evidence(grid_signals: list[dict], price_signals: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in grid_signals + price_signals:
        out.extend(item.get("evidence", []))
    return out


def _signal_present(names: list[str], candidates: set[str]) -> bool:
    return any(name in candidates for name in names)


def _driver_summary(drivers: list[str], fallback: str) -> str:
    return " / ".join(drivers) if drivers else fallback


def reason_about_market(grid_signals: list[dict], price_signals: list[dict]) -> dict:
    grid_names = _names(grid_signals, "signal_name")
    price_names = _names(price_signals, "price_signal")
    evidence = _evidence(grid_signals, price_signals)

    has_load_up = "demand spike" in grid_names
    has_load_down = "demand drop" in grid_names
    has_wind_drop = "wind generation drop" in grid_names
    has_solar_drop = "solar generation drop" in grid_names
    has_tightening = "possible supply tightening" in grid_names
    has_renewable_recovery = "renewable generation recovery" in grid_names
    has_price_spike = _signal_present(price_names, {"real-time price spike", "real-time premium to day-ahead"})
    has_price_drop = _signal_present(price_names, {"real-time price drop", "negative price signal"})
    has_da_spread = "real-time premium to day-ahead" in price_names
    has_da_discount = "real-time discount to day-ahead" in price_names
    has_volatility = "real-time price volatility" in price_names

    base_unmodeled = [
        "localized congestion or transmission constraint",
        "generator outage or derate",
        "ancillary service or reserve condition",
        "local topology and settlement-point congestion effects",
    ]
    caveats = [
        "ERCOT grid and price data are observational; this workflow does not prove causality.",
        "Transmission constraints, outages, ancillary conditions, or local congestion may affect prices but are not fully modeled in v1.",
    ]

    if (has_tightening or has_load_up or has_wind_drop or has_solar_drop) and has_price_spike:
        drivers = []
        observations = ["real-time prices moved higher or cleared above day-ahead expectations"]
        if has_load_up:
            drivers.append("higher demand")
            observations.append("load is above its recent rolling baseline")
        if has_wind_drop or has_solar_drop:
            drivers.append("lower renewable output")
            observations.append("wind or solar output is below its recent rolling baseline")
        if has_da_spread:
            observations.append("real-time prices are materially above the matching day-ahead price")
        market_read = f"Real-time prices are showing upward pressure while visible grid conditions point to {_driver_summary(drivers, 'physical grid tightening')}."
        explanation = f"Grid conditions and real-time pricing moved in the same direction. The evidence points to {' and '.join(drivers) or 'physical grid tightening'} coinciding with upward real-time price pressure."
        return ReasoningOutput(
            explanation=explanation,
            likely_driver=_driver_summary(drivers, "supply tightening / demand-driven price pressure"),
            market_read=market_read,
            likely_drivers=drivers or ["supply tightening / demand-driven price pressure"],
            supporting_observations=observations,
            unmodeled_factors=base_unmodeled,
            confidence=CONFIDENCE_HIGH,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_renewable_recovery and has_price_drop:
        observations = [
            "renewable output is recovering relative to its recent rolling baseline",
            "real-time prices weakened or turned negative during the observed interval",
        ]
        return ReasoningOutput(
            explanation="Renewable output recovered while real-time prices weakened or turned negative, which is consistent with surplus renewable pressure during the observed interval.",
            likely_driver="renewable recovery / surplus energy pressure",
            market_read="Real-time prices weakened while renewable output recovered, which is consistent with surplus energy pressure.",
            likely_drivers=["renewable recovery", "surplus energy pressure"],
            supporting_observations=observations,
            unmodeled_factors=["local congestion", "curtailment conditions", "negative-price settlement point effects"],
            confidence=CONFIDENCE_HIGH,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_da_discount and (has_load_down or has_renewable_recovery or has_price_drop):
        drivers = []
        observations = ["real-time prices are materially below the matching day-ahead price"]
        if has_load_down:
            drivers.append("lower real-time load")
            observations.append("load is below its recent rolling baseline")
        if has_renewable_recovery:
            drivers.append("stronger renewable output")
            observations.append("wind or solar output is above its recent rolling baseline")
        if has_price_drop:
            observations.append("real-time prices weakened relative to the recent rolling baseline")
        return ReasoningOutput(
            explanation="Real-time prices cleared below day-ahead while load and renewable signals point to looser real-time conditions than the day-ahead market expected.",
            likely_driver=_driver_summary(drivers, "looser real-time conditions versus day-ahead expectations"),
            market_read="Real-time prices are below day-ahead while visible grid conditions look looser, which is consistent with actual supply-demand conditions coming in easier than day-ahead expectations.",
            likely_drivers=drivers or ["looser real-time conditions versus day-ahead expectations"],
            supporting_observations=observations,
            unmodeled_factors=[
                "weather and temperature forecast error",
                "day-ahead load forecast error",
                "renewable forecast error",
                "day-ahead congestion or commitment assumptions that did not persist in real time",
            ],
            confidence=CONFIDENCE_MEDIUM,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_da_discount:
        return ReasoningOutput(
            explanation="Real-time prices cleared materially below day-ahead, but the current v1 grid signals do not fully explain the discount.",
            likely_driver="looser real-time conditions versus day-ahead expectations",
            market_read="Real-time prices are below day-ahead expectations. This can happen when actual load, renewable output, congestion, or commitment conditions are easier than expected, but the visible v1 signals only partially explain the move.",
            likely_drivers=["looser real-time conditions versus day-ahead expectations"],
            supporting_observations=["real-time prices are materially below the matching day-ahead price"],
            unmodeled_factors=[
                "weather and temperature forecast error",
                "day-ahead load forecast error",
                "renewable forecast error",
                "day-ahead congestion or commitment assumptions that did not persist in real time",
            ],
            confidence=CONFIDENCE_MEDIUM,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_da_spread:
        observations = ["real-time prices diverged materially from the matching day-ahead price"]
        if has_volatility:
            observations.append("real-time prices are volatile over the recent interval window")
        return ReasoningOutput(
            explanation="Real-time prices diverged materially from day-ahead pricing. This suggests a real-time imbalance relative to the day-ahead expectation, but the available v1 grid signals do not fully explain the spread.",
            likely_driver="real-time imbalance versus day-ahead schedule",
            market_read="Real-time prices are materially different from day-ahead expectations, but the visible system-wide grid signals only partially explain the move.",
            likely_drivers=["real-time imbalance versus day-ahead schedule", "localized congestion or constraint"],
            supporting_observations=observations,
            unmodeled_factors=base_unmodeled,
            confidence=CONFIDENCE_MEDIUM,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if has_price_spike:
        observations = ["real-time prices increased without a matching system-wide load or renewable signal in this v1 dataset"]
        if has_volatility:
            observations.append("recent real-time prices are volatile")
        return ReasoningOutput(
            explanation="Real-time prices increased without a strongly aligned load or renewable signal. A constraint, outage, reserve condition, or localized congestion could be involved, but confidence is low in v1.",
            likely_driver="possible constraint or outage",
            market_read="Real-time prices increased, but the currently modeled grid signals do not provide a complete explanation.",
            likely_drivers=["localized congestion or constraint", "generator outage or derate", "reserve or scarcity condition"],
            supporting_observations=observations,
            unmodeled_factors=base_unmodeled,
            confidence=CONFIDENCE_LOW,
            evidence=evidence,
            caveats=caveats,
        ).__dict__

    if not evidence:
        return ReasoningOutput(
            explanation="insufficient data",
            likely_driver="insufficient data",
            market_read="There is not enough ERCOT data in the selected window to form a market read.",
            likely_drivers=["insufficient data"],
            supporting_observations=[],
            unmodeled_factors=[],
            confidence=CONFIDENCE_LOW,
            evidence=[],
            caveats=["Required ERCOT datasets were missing or too sparse for rule-based interpretation."],
        ).__dict__

    return ReasoningOutput(
        explanation="Observed grid and price movements do not show a strong aligned signal in the selected window.",
        likely_driver="no material aligned signal",
        market_read="The selected window does not show a strong aligned grid-to-price signal.",
        likely_drivers=["no material aligned signal"],
        supporting_observations=["available grid and price signals are not strongly aligned"],
        unmodeled_factors=base_unmodeled,
        confidence=CONFIDENCE_LOW,
        evidence=evidence,
        caveats=caveats,
    ).__dict__
