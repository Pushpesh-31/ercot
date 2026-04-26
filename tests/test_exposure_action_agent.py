from src.agents.exposure_action_agent import generate_exposure_actions


def test_exposure_actions_reference_reasoning_drivers():
    reasoning = {
        "likely_drivers": ["localized congestion or constraint", "generator outage or derate"],
        "likely_driver": "possible constraint or outage",
    }
    price = [{"price_signal": "real-time price spike"}]
    grid = [{"signal_name": "no material grid signal"}]

    result = generate_exposure_actions("market observer", grid, price, reasoning)

    assert "localized congestion or constraint" in result["exposure"]
    assert any("congestion" in action for action in result["possible_actions"])


def test_industrial_consumer_discount_actions_do_not_reference_premiums():
    reasoning = {
        "likely_drivers": ["lower real-time load", "stronger renewable output"],
        "likely_driver": "looser real-time conditions versus day-ahead expectations",
    }
    price = [{"price_signal": "real-time discount to day-ahead"}]
    grid = [{"signal_name": "demand drop"}]

    result = generate_exposure_actions("industrial consumer", grid, price, reasoning)

    assert "favorable versus day-ahead" in result["exposure"]
    assert not any("premium" in action for action in result["possible_actions"])
