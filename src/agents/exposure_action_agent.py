from __future__ import annotations

from src.utils.schemas import DISCLAIMER, ExposureOutput


PERSONAS = ["industrial consumer", "generator / asset owner", "market observer"]


def _material_signals(grid_signals: list[dict], price_signals: list[dict]) -> list[str]:
    names = []
    for signal in grid_signals:
        name = signal.get("signal_name")
        if name and name not in {"no material grid signal", "insufficient data"}:
            names.append(name)
    for signal in price_signals:
        name = signal.get("price_signal")
        if name and name not in {"no material price signal", "insufficient data"}:
            names.append(name)
    return names


def _driver_text(reasoning: dict) -> str:
    drivers = reasoning.get("likely_drivers") or [reasoning.get("likely_driver")]
    clean = [driver for driver in drivers if driver and driver != "insufficient data"]
    return ", ".join(clean) if clean else "the observed market signals"


def _is_loose_condition(reasoning: dict, price_names: list[str]) -> bool:
    drivers = " ".join(reasoning.get("likely_drivers", []) + [reasoning.get("likely_driver", "")])
    return "real-time discount to day-ahead" in price_names or "looser real-time conditions" in drivers


def generate_exposure_actions(persona: str, grid_signals: list[dict], price_signals: list[dict], reasoning: dict) -> dict:
    linked = _material_signals(grid_signals, price_signals)
    if not linked:
        return ExposureOutput(
            persona=persona,
            exposure="insufficient data or no material exposure signal detected in the selected window",
            possible_actions=["monitor subsequent ERCOT intervals", "compare the next refresh with day-ahead prices"],
            linked_signals=[],
            disclaimer=DISCLAIMER,
        ).__dict__

    price_names = [signal.get("price_signal", "") for signal in price_signals]
    upward = any(name in price_names for name in ["real-time price spike", "real-time premium to day-ahead", "real-time price volatility"])
    negative = "negative price signal" in price_names or "real-time price drop" in price_names
    loose = _is_loose_condition(reasoning, price_names)
    driver_text = _driver_text(reasoning)
    has_unmodeled_driver = any(driver in driver_text for driver in ["congestion", "constraint", "outage", "reserve", "scarcity"])

    if persona == "industrial consumer":
        if loose:
            exposure = f"for load with real-time-indexed or pass-through exposure, costs may be favorable versus day-ahead, tied to {driver_text}. fixed or hedged supply may see limited direct impact unless the contract passes through index, congestion, ancillary, imbalance, or settlement charges"
            actions = [
                "compare actual real-time exposure against day-ahead, fixed-price, or hedged coverage for the same interval",
                "watch whether the RT discount persists or reverses in the next refresh",
                "check whether lower load or stronger renewable output continues to support the discount",
                "review supply contract pass-through language for congestion, ancillary, imbalance, and settlement charges",
                "treat weather/load forecast error and renewable forecast error as plausible day-ahead drivers, not confirmed causes",
            ]
        else:
            exposure = (
                f"for load with real-time-indexed or pass-through exposure, costs may be elevated, tied to {driver_text}. fixed or hedged supply may mute the direct impact unless the contract passes through index, congestion, ancillary, imbalance, or settlement charges"
                if upward
                else f"industrial cost impact depends on contract structure; any index or pass-through exposure may vary with {driver_text}, while fixed or hedged supply may see limited direct impact"
            )
            actions = [
                "compare current interval exposure against day-ahead, fixed-price, or hedged coverage",
                "review supply contract pass-through language for congestion, ancillary, imbalance, and settlement charges",
                "check whether flexible load has value if real-time premiums persist",
                "watch whether the same settlement point continues to clear above day-ahead in the next refresh" if upward else "watch whether the observed price condition persists in the next refresh",
                "review congestion, outage, and reserve reports before attributing the move to demand alone" if has_unmodeled_driver else "watch load and renewable trends to see whether the driver persists",
            ]
    elif persona == "generator / asset owner":
        if upward:
            exposure = f"higher selected settlement-point prices may improve revenue opportunity for available generation exposed to that price, tied to {driver_text}. asset-specific value depends on node, marginal cost, availability, hedge position, and dispatch constraints"
            actions = [
                "compare the selected settlement point with the asset node or settlement exposure",
                "compare market price against marginal operating cost, fuel cost, start cost, and offer assumptions",
                "check availability, outage status, and dispatch constraints before treating the price move as realizable value",
                "review hedge or forward-sale exposure to see whether higher spot prices flow through to revenue",
                "separate hub-wide movement from localized nodal or congestion effects",
            ]
        else:
            exposure = f"lower or discounted real-time prices may pressure revenue for exposed generation, tied to {driver_text}. asset-specific impact depends on node, marginal cost, availability, hedge position, and dispatch constraints"
            actions = [
                "compare the selected settlement point with the asset node or settlement exposure",
                "check whether the asset remains economic against marginal operating cost and offer assumptions",
                "review hedge or forward-sale exposure to see whether lower spot prices are offset contractually",
                "track whether load or renewable conditions continue to support the lower-price signal",
                "separate hub-wide movement from localized nodal or congestion effects",
            ]
    else:
        exposure = f"market movement appears tied to {driver_text}" if upward else f"market variability appears tied to {driver_text}" if negative or loose else f"market variability with {driver_text}"
        actions = [
            "compare the next refresh against the current market read",
            "track whether real-time prices continue to diverge from day-ahead",
            "watch load, wind, and solar baselines for confirmation or reversal",
            "check forecast error, congestion, and commitment assumptions before attributing the RT-to-DA gap to one cause",
        ]

    return ExposureOutput(
        persona=persona,
        exposure=exposure,
        possible_actions=actions,
        linked_signals=linked,
        disclaimer=DISCLAIMER,
    ).__dict__
