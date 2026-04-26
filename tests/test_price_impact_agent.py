import pandas as pd

from src.agents.price_impact_agent import detect_price_signals


def test_detects_rt_price_spike_and_da_premium():
    ts = pd.date_range("2026-04-24 00:00", periods=6, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame({"timestamp": ts, "price": [20, 20, 20, 20, 20, 60]})
    da = pd.DataFrame({"timestamp": [ts[-1].floor("h")], "price": [25]})

    signals = detect_price_signals(rt, da)
    names = {signal["price_signal"] for signal in signals}

    assert "real-time price spike" in names
    assert "real-time premium to day-ahead" in names


def test_detects_negative_price():
    ts = pd.date_range("2026-04-24 00:00", periods=6, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame({"timestamp": ts, "price": [10, 10, 10, 10, 10, -5]})

    signals = detect_price_signals(rt, pd.DataFrame())
    assert any(signal["price_signal"] == "negative price signal" for signal in signals)


def test_detects_rt_discount_to_day_ahead():
    ts = pd.date_range("2026-04-24 00:00", periods=6, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame({"timestamp": ts, "price": [30, 30, 30, 30, 30, 20]})
    da = pd.DataFrame({"timestamp": [ts[-1].floor("h")], "price": [45]})

    signals = detect_price_signals(rt, da)

    assert any(signal["price_signal"] == "real-time discount to day-ahead" for signal in signals)
