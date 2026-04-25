import pandas as pd

from src.agents.grid_signal_agent import detect_grid_signals


def test_detects_supply_tightening_when_load_up_and_wind_down():
    ts = pd.date_range("2026-04-24 00:00", periods=6, freq="h", tz="America/Chicago")
    load = pd.DataFrame({"timestamp": ts, "load_mw": [100, 100, 100, 100, 100, 108]})
    wind = pd.DataFrame({"timestamp": ts, "wind_mw": [100, 100, 100, 100, 100, 80]})
    solar = pd.DataFrame({"timestamp": ts, "solar_mw": [50, 50, 50, 50, 50, 50]})

    signals = detect_grid_signals(load, wind, solar)
    names = {signal["signal_name"] for signal in signals}

    assert "demand spike" in names
    assert "wind generation drop" in names
    assert "possible supply tightening" in names


def test_insufficient_grid_data_when_empty():
    signals = detect_grid_signals(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert signals[0]["signal_name"] == "insufficient data"
