import pandas as pd

from src.agents.persona_exposure_agent import build_agent_brief, build_market_pulse, build_persona_exposures


def test_market_pulse_flags_high_risk_for_price_premium_and_tight_grid():
    ts = pd.date_range("2026-04-24 00:00", periods=6, freq="h", tz="America/Chicago")
    rt = pd.DataFrame({"timestamp": ts, "price": [30, 35, 40, 45, 50, 180]})
    da = pd.DataFrame({"timestamp": [ts[-1]], "price": [65]})
    load = pd.DataFrame({"timestamp": ts, "load_mw": [1000, 1000, 1000, 1000, 1000, 1125]})
    wind = pd.DataFrame({"timestamp": ts, "wind_mw": [300, 300, 300, 300, 300, 210]})
    solar = pd.DataFrame({"timestamp": ts, "solar_mw": [100, 100, 100, 100, 100, 100]})

    pulse = build_market_pulse(
        rt,
        da,
        load,
        wind,
        solar,
        [{"signal_name": "possible supply tightening"}, {"signal_name": "demand spike"}],
        [{"price_signal": "real-time premium to day-ahead"}],
    )

    assert pulse["risk_label"] == "High"
    assert pulse["rt_da_spread"] == 115
    assert pulse["renewable_share"] > 0


def test_persona_cards_render_for_partial_data():
    pulse = build_market_pulse(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        [{"signal_name": "insufficient data"}],
        [{"price_signal": "insufficient data"}],
    )

    cards = build_persona_exposures(pulse, [], [], {"likely_drivers": []})

    assert len(cards) == 6
    assert {card["persona"] for card in cards} == {
        "Data center operator",
        "Industrial energy buyer",
        "Power trader",
        "Retail electricity provider",
        "Renewable operator",
        "Battery/storage operator",
    }
    assert all(card["assumptions"] for card in cards)


def test_agent_brief_identifies_most_exposed_personas():
    pulse = {"risk_label": "High", "rt_da_spread": 90, "confidence": "Medium"}
    cards = [
        {"persona": "Power trader", "exposure_level": "High"},
        {"persona": "Industrial energy buyer", "exposure_level": "Medium"},
    ]

    brief = build_agent_brief(pulse, cards, {"likely_drivers": ["real-time premium to day-ahead"]})

    assert brief["current_market_risk"] == "High"
    assert brief["most_exposed_personas"] == ["Power trader"]
    assert "real-time premium to day-ahead" in brief["main_drivers"]
