from src.agents.reasoning_agent import reason_about_market


def test_high_confidence_when_grid_and_price_align():
    grid = [{"signal_name": "demand spike", "evidence": [{"metric": "load"}]}]
    price = [{"price_signal": "real-time price spike", "evidence": [{"metric": "price"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "High"
    assert "demand" in result["likely_driver"]
    assert "higher demand" in result["likely_drivers"]
    assert result["market_read"]
    assert result["supporting_observations"]


def test_low_confidence_when_price_spike_unexplained():
    grid = [{"signal_name": "no material grid signal", "evidence": []}]
    price = [{"price_signal": "real-time price spike", "evidence": [{"metric": "price"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "Low"
    assert result["likely_driver"] == "possible constraint or outage"
    assert "localized congestion or constraint" in result["likely_drivers"]
    assert "generator outage or derate" in result["likely_drivers"]


def test_medium_confidence_when_rt_premium_not_explained_by_grid():
    grid = [{"signal_name": "no material grid signal", "evidence": []}]
    price = [{"price_signal": "real-time premium to day-ahead", "evidence": [{"metric": "spread"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "Medium"
    assert "real-time imbalance versus day-ahead schedule" in result["likely_drivers"]
    assert result["unmodeled_factors"]


def test_high_confidence_when_renewables_recover_and_prices_drop():
    grid = [{"signal_name": "renewable generation recovery", "evidence": [{"metric": "wind"}]}]
    price = [{"price_signal": "negative price signal", "evidence": [{"metric": "price"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "High"
    assert "surplus energy pressure" in result["likely_drivers"]


def test_rt_discount_with_lower_load_points_to_looser_conditions():
    grid = [
        {"signal_name": "demand drop", "evidence": [{"metric": "load"}]},
        {"signal_name": "renewable generation recovery", "evidence": [{"metric": "wind"}]},
    ]
    price = [{"price_signal": "real-time discount to day-ahead", "evidence": [{"metric": "spread"}]}]

    result = reason_about_market(grid, price)

    assert result["confidence"] == "Medium"
    assert "lower real-time load" in result["likely_drivers"]
    assert "stronger renewable output" in result["likely_drivers"]
    assert "weather and temperature forecast error" in result["unmodeled_factors"]
