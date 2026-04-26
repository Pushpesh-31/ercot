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
