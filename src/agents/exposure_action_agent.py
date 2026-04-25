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

    if persona == "industrial consumer":
        exposure = "real-time power cost risk" if upward else "potential variability in real-time indexed power costs"
        actions = [
            "evaluate flexible load reduction where operationally feasible",
            "review exposure to real-time indexed pricing",
            "consider hedging future exposure",
            "monitor volatility in upcoming settlement intervals",
        ]
    elif persona == "generator / asset owner":
        exposure = "potential revenue opportunity or operational dispatch signal" if upward else "potential revenue pressure during weaker price intervals"
        actions = [
            "monitor dispatch economics",
            "evaluate availability during high-price periods" if upward else "review availability assumptions during low-price periods",
            "review outage timing against observed volatility",
            "monitor renewable recovery and load trend",
        ]
    else:
        exposure = "market volatility / grid stress" if upward else "market variability with possible surplus energy pressure" if negative else "market variability"
        actions = [
            "monitor next intervals",
            "compare with day-ahead prices",
            "watch renewable recovery and load trend",
            "review whether the same driver persists after the next hourly refresh",
        ]

    return ExposureOutput(
        persona=persona,
        exposure=exposure,
        possible_actions=actions,
        linked_signals=linked,
        disclaimer=DISCLAIMER,
    ).__dict__
