from __future__ import annotations

import pandas as pd

from src.utils.schemas import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, GridSignal, INSUFFICIENT_DATA
from src.utils.time_utils import iso_ts


def _latest_and_baseline(df: pd.DataFrame, value_col: str) -> tuple[pd.Series | None, float | None]:
    if df.empty or value_col not in df.columns or "timestamp" not in df.columns:
        return None, None
    clean = df.dropna(subset=[value_col]).sort_values("timestamp")
    if len(clean) < 2:
        return None, None
    latest = clean.iloc[-1]
    baseline = clean.iloc[:-1][value_col].tail(min(8, len(clean) - 1)).mean()
    if pd.isna(baseline) or baseline == 0:
        return latest, None
    return latest, float(baseline)


def _pct_change(current: float, baseline: float | None) -> float | None:
    if baseline in (None, 0):
        return None
    return (current - baseline) / baseline


def _evidence(metric: str, observed, baseline, timestamp, source: str) -> dict:
    return {
        "metric": metric,
        "observed_value": round(float(observed), 2) if observed is not None and pd.notna(observed) else INSUFFICIENT_DATA,
        "baseline": round(float(baseline), 2) if baseline is not None and pd.notna(baseline) else INSUFFICIENT_DATA,
        "timestamp": iso_ts(timestamp),
        "source_dataset": source,
    }


def detect_grid_signals(load_df: pd.DataFrame, wind_df: pd.DataFrame, solar_df: pd.DataFrame) -> list[dict]:
    signals: list[GridSignal] = []

    load_latest, load_baseline = _latest_and_baseline(load_df, "load_mw")
    wind_latest, wind_baseline = _latest_and_baseline(wind_df, "wind_mw")
    solar_latest, solar_baseline = _latest_and_baseline(solar_df, "solar_mw")

    load_change = None
    if load_latest is not None and load_baseline:
        load_change = _pct_change(float(load_latest["load_mw"]), load_baseline)
        if load_change is not None and abs(load_change) > 0.05:
            signals.append(
                GridSignal(
                    signal_name="demand spike" if load_change > 0 else "demand drop",
                    direction="up" if load_change > 0 else "down",
                    magnitude=round(load_change * 100, 2),
                    timestamp=iso_ts(load_latest["timestamp"]),
                    evidence=[
                        _evidence(
                            "ERCOT load",
                            load_latest["load_mw"],
                            load_baseline,
                            load_latest["timestamp"],
                            "Load forecast / demand data",
                        ),
                    ],
                    confidence=CONFIDENCE_MEDIUM,
                ),
            )

    wind_change = None
    if wind_latest is not None and wind_baseline:
        wind_change = _pct_change(float(wind_latest["wind_mw"]), wind_baseline)
        if wind_change is not None and abs(wind_change) > 0.10:
            signals.append(
                GridSignal(
                    signal_name="wind generation drop" if wind_change < 0 else "renewable generation recovery",
                    direction="down" if wind_change < 0 else "up",
                    magnitude=round(wind_change * 100, 2),
                    timestamp=iso_ts(wind_latest["timestamp"]),
                    evidence=[
                        _evidence(
                            "ERCOT wind generation",
                            wind_latest["wind_mw"],
                            wind_baseline,
                            wind_latest["timestamp"],
                            "NP4-732-CD wind actual and forecast",
                        ),
                    ],
                    confidence=CONFIDENCE_MEDIUM,
                ),
            )

    solar_change = None
    if solar_latest is not None and solar_baseline:
        hour = pd.Timestamp(solar_latest["timestamp"]).hour
        solar_change = _pct_change(float(solar_latest["solar_mw"]), solar_baseline)
        daylight = 7 <= hour <= 19
        if daylight and solar_change is not None and abs(solar_change) > 0.10:
            signals.append(
                GridSignal(
                    signal_name="solar generation drop" if solar_change < 0 else "renewable generation recovery",
                    direction="down" if solar_change < 0 else "up",
                    magnitude=round(solar_change * 100, 2),
                    timestamp=iso_ts(solar_latest["timestamp"]),
                    evidence=[
                        _evidence(
                            "ERCOT solar generation",
                            solar_latest["solar_mw"],
                            solar_baseline,
                            solar_latest["timestamp"],
                            "NP4-737-CD solar actual and forecast",
                        ),
                    ],
                    confidence=CONFIDENCE_MEDIUM,
                ),
            )

    renewables_down = any(change is not None and change < -0.10 for change in [wind_change, solar_change])
    if load_change is not None and load_change > 0.05 and renewables_down:
        evidence = []
        if load_latest is not None:
            evidence.append(_evidence("ERCOT load", load_latest["load_mw"], load_baseline, load_latest["timestamp"], "Load forecast / demand data"))
        if wind_latest is not None and wind_change is not None and wind_change < -0.10:
            evidence.append(_evidence("ERCOT wind generation", wind_latest["wind_mw"], wind_baseline, wind_latest["timestamp"], "NP4-732-CD wind actual and forecast"))
        if solar_latest is not None and solar_change is not None and solar_change < -0.10:
            evidence.append(_evidence("ERCOT solar generation", solar_latest["solar_mw"], solar_baseline, solar_latest["timestamp"], "NP4-737-CD solar actual and forecast"))
        signals.append(
            GridSignal(
                signal_name="possible supply tightening",
                direction="tightening",
                magnitude=None,
                timestamp=evidence[0]["timestamp"] if evidence else "",
                evidence=evidence,
                confidence=CONFIDENCE_HIGH,
            ),
        )

    if not signals:
        signals.append(
            GridSignal(
                signal_name=INSUFFICIENT_DATA if any(df.empty for df in [load_df, wind_df, solar_df]) else "no material grid signal",
                direction="flat",
                magnitude=None,
                timestamp="",
                evidence=[],
                confidence=CONFIDENCE_LOW,
            ),
        )

    return [signal.__dict__ for signal in signals]
