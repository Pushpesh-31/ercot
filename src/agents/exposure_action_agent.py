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
    driver_text = _driver_text(reasoning)
    has_unmodeled_driver = any(driver in driver_text for driver in ["congestion", "constraint", "outage", "reserve", "scarcity"])

    if persona == "industrial consumer":
        exposure = f"real-time indexed cost exposure tied to {driver_text}" if upward else f"potential variability in indexed power costs tied to {driver_text}"
        actions = [
            "compare current interval exposure against day-ahead or fixed-price coverage",
            "check whether flexible load can avoid intervals with persistent real-time premiums",
            "watch whether the same settlement point continues to clear above day-ahead in the next refresh",
            "review congestion, outage, and reserve reports before attributing the move to demand alone" if has_unmodeled_driver else "watch load and renewable trends to see whether the driver persists",
        ]
    elif persona == "generator / asset owner":
        exposure = f"dispatch economics are being influenced by {driver_text}" if upward else f"revenue conditions may be pressured by {driver_text}"
        actions = [
            "compare unit availability and offer assumptions against the observed settlement-point move",
            "monitor whether the price signal persists beyond a single interval",
            "review local congestion, outage, and reserve context before treating the move as system-wide" if has_unmodeled_driver else "track whether load or renewable conditions continue to support the price move",
            "separate hub-wide movement from localized settlement-point effects",
        ]
    else:
        exposure = f"market movement appears tied to {driver_text}" if upward else f"market variability appears tied to {driver_text}" if negative else f"market variability with {driver_text}"
        actions = [
            "compare the next refresh against the current market read",
            "track whether real-time prices continue to diverge from day-ahead",
            "watch load, wind, and solar baselines for confirmation or reversal",
            "treat congestion, outage, and reserve conditions as unmodeled explanations when system-wide grid signals do not explain price moves",
        ]

    return ExposureOutput(
        persona=persona,
        exposure=exposure,
        possible_actions=actions,
        linked_signals=linked,
        disclaimer=DISCLAIMER,
    ).__dict__
