from src.agents.reasoning_agent import reason_about_market


def test_high_confidence_when_grid_and_price_align():
    grid = [{"signal_name": "demand spike", "evidence": [{"metric": "load"}]}]
    price = [{"price_signal": "real-time price spike", "evidence": [{"metric": "price"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "High"
    assert "demand" in result["likely_driver"]


def test_low_confidence_when_price_spike_unexplained():
    grid = [{"signal_name": "no material grid signal", "evidence": []}]
    price = [{"price_signal": "real-time price spike", "evidence": [{"metric": "price"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "Low"
    assert result["likely_driver"] == "possible constraint or outage"
