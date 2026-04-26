from __future__ import annotations

import pandas as pd

from src.utils.schemas import CONFIDENCE_LOW, CONFIDENCE_MEDIUM, INSUFFICIENT_DATA, PriceSignal
from src.utils.time_utils import iso_ts


def _latest_da(da_df: pd.DataFrame, ts) -> float | None:
    if da_df.empty or "price" not in da_df.columns or "timestamp" not in da_df.columns:
        return None
    target = pd.Timestamp(ts).floor("h")
    df = da_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    exact = df[df["timestamp"].dt.floor("h") == target]
    if exact.empty:
        return None
    return float(exact.sort_values("timestamp").iloc[-1]["price"])


def _evidence(metric: str, observed, baseline, timestamp, source: str) -> dict:
    return {
        "metric": metric,
        "observed_value": round(float(observed), 2) if observed is not None and pd.notna(observed) else INSUFFICIENT_DATA,
        "baseline": round(float(baseline), 2) if baseline is not None and pd.notna(baseline) else INSUFFICIENT_DATA,
        "timestamp": iso_ts(timestamp),
        "source_dataset": source,
    }


def detect_price_signals(rt_df: pd.DataFrame, da_df: pd.DataFrame) -> list[dict]:
    if rt_df.empty or "price" not in rt_df.columns or "timestamp" not in rt_df.columns:
        return [
            PriceSignal(
                price_signal=INSUFFICIENT_DATA,
                price_change=None,
                current_price=None,
                baseline_price=None,
                day_ahead_price=None,
                rt_da_spread=None,
                evidence=[],
                confidence=CONFIDENCE_LOW,
            ).__dict__,
        ]

    clean = rt_df.dropna(subset=["price"]).sort_values("timestamp")
    if len(clean) < 2:
        return [
            PriceSignal(
                price_signal=INSUFFICIENT_DATA,
                price_change=None,
                current_price=float(clean.iloc[-1]["price"]) if len(clean) else None,
                baseline_price=None,
                day_ahead_price=None,
                rt_da_spread=None,
                evidence=[],
                confidence=CONFIDENCE_LOW,
            ).__dict__,
        ]

    latest = clean.iloc[-1]
    current = float(latest["price"])
    baseline = float(clean.iloc[:-1]["price"].tail(min(16, len(clean) - 1)).mean())
    price_change = None if baseline == 0 else (current - baseline) / abs(baseline)
    da_price = _latest_da(da_df, latest["timestamp"])
    spread = current - da_price if da_price is not None else None
    rolling_std = float(clean["price"].tail(min(16, len(clean))).std())
    median_price = float(clean["price"].tail(min(16, len(clean))).median())

    signals: list[PriceSignal] = []
    common = {
        "current_price": round(current, 2),
        "baseline_price": round(baseline, 2),
        "day_ahead_price": round(da_price, 2) if da_price is not None else None,
        "rt_da_spread": round(spread, 2) if spread is not None else None,
    }

    if price_change is not None and abs(price_change) > 0.50:
        signals.append(
            PriceSignal(
                price_signal="real-time price spike" if price_change > 0 else "real-time price drop",
                price_change=round(price_change * 100, 2),
                evidence=[_evidence("Real-time settlement point price", current, baseline, latest["timestamp"], "NP6-905-CD real-time settlement point prices")],
                confidence=CONFIDENCE_MEDIUM,
                **common,
            ),
        )

    if current < 0:
        signals.append(
            PriceSignal(
                price_signal="negative price signal",
                price_change=round(price_change * 100, 2) if price_change is not None else None,
                evidence=[_evidence("Real-time settlement point price", current, 0, latest["timestamp"], "NP6-905-CD real-time settlement point prices")],
                confidence=CONFIDENCE_MEDIUM,
                **common,
            ),
        )

    if da_price is not None and current > 2 * da_price and current > da_price:
        signals.append(
            PriceSignal(
                price_signal="real-time premium to day-ahead",
                price_change=round(price_change * 100, 2) if price_change is not None else None,
                evidence=[_evidence("RT minus DA spread", spread, da_price, latest["timestamp"], "NP6-905-CD and NP4-190-CD settlement point prices")],
                confidence=CONFIDENCE_MEDIUM,
                **common,
            ),
        )

    if da_price is not None and spread is not None and current < da_price:
        discount_threshold = max(10.0, abs(da_price) * 0.25)
        if abs(spread) > discount_threshold:
            signals.append(
                PriceSignal(
                    price_signal="real-time discount to day-ahead",
                    price_change=round(price_change * 100, 2) if price_change is not None else None,
                    evidence=[_evidence("RT minus DA spread", spread, da_price, latest["timestamp"], "NP6-905-CD and NP4-190-CD settlement point prices")],
                    confidence=CONFIDENCE_MEDIUM,
                    **common,
                ),
            )

    volatility_threshold = max(25.0, abs(median_price) * 0.75)
    if rolling_std > volatility_threshold:
        signals.append(
            PriceSignal(
                price_signal="real-time price volatility",
                price_change=round(price_change * 100, 2) if price_change is not None else None,
                evidence=[_evidence("Rolling RT price standard deviation", rolling_std, volatility_threshold, latest["timestamp"], "NP6-905-CD real-time settlement point prices")],
                confidence=CONFIDENCE_MEDIUM,
                **common,
            ),
        )

    if not signals:
        signals.append(
            PriceSignal(
                price_signal="no material price signal",
                price_change=round(price_change * 100, 2) if price_change is not None else None,
                evidence=[_evidence("Real-time settlement point price", current, baseline, latest["timestamp"], "NP6-905-CD real-time settlement point prices")],
                confidence=CONFIDENCE_LOW,
                **common,
            ),
        )

    return [signal.__dict__ for signal in signals]
