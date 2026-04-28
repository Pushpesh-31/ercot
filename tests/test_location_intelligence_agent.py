import pandas as pd

from src.agents.location_intelligence_agent import (
    build_location_intelligence,
    classify_settlement_point,
    normalize_settlement_point_mapping,
    price_spike_leaderboards,
    spread_leaderboard,
    top_location_explanations,
    volatility_leaderboard,
)


def test_classifies_settlement_point_types():
    assert classify_settlement_point("HB_HOUSTON") == "hub"
    assert classify_settlement_point("LZ_NORTH") == "load_zone"
    assert classify_settlement_point("RNODE_1") == "resource_node"


def test_build_location_intelligence_scores_congestion_and_spread():
    ts = pd.date_range("2026-04-24 00:00", periods=8, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame(
        {
            "timestamp": list(ts) * 3,
            "settlement_point": ["HB_HOUSTON"] * 8 + ["LZ_NORTH"] * 8 + ["RNODE_A"] * 8,
            "rt_price": [25] * 8 + [30] * 8 + [35, 40, 45, 50, 55, 90, 140, 190],
        },
    )
    da = pd.DataFrame(
        {
            "timestamp": [ts[-1].floor("h")] * 3,
            "settlement_point": ["HB_HOUSTON", "LZ_NORTH", "RNODE_A"],
            "da_price": [25, 30, 60],
        },
    )

    result = build_location_intelligence(rt, da, volatility_window=4)
    latest_node = result[result["settlement_point"] == "RNODE_A"].sort_values("timestamp").iloc[-1]

    assert "deviation_from_hub_or_zone_average" in result.columns
    assert latest_node["rt_da_spread"] == 130
    assert latest_node["congestion_label"] == "High congestion"
    assert latest_node["volatility_label"] in {"Volatile", "Extreme"}


def test_location_intelligence_joins_mapping_and_compares_to_reference_hub():
    ts = pd.date_range("2026-04-24 00:00", periods=2, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame(
        {
            "timestamp": [ts[-1], ts[-1]],
            "settlement_point": ["HB_HOUSTON", "MUSTNGCK_ALL"],
            "rt_price": [40, 95],
        },
    )
    mapping = pd.DataFrame(
        {
            "Resource Node": ["MUSTNGCK_ALL"],
            "Settlement Load Zone": ["LZ_HOUSTON"],
            "Hub": ["HB_HOUSTON"],
        },
    )

    result = build_location_intelligence(rt, pd.DataFrame(), mapping)
    node = result[result["settlement_point"] == "MUSTNGCK_ALL"].iloc[0]

    assert node["reference_hub"] == "HB_HOUSTON"
    assert node["reference_hub_rt_price"] == 40
    assert node["deviation_from_reference_hub"] == 55
    assert node["mapping_confidence"] == "Official mapping"
    assert node["settlement_point_display"] == "MUSTNGCK_ALL · HB_HOUSTON"


def test_normalize_mapping_falls_back_to_load_zone_when_hub_missing():
    mapping = pd.DataFrame(
        {
            "Resource Node": ["NODE_A"],
            "Settlement Load Zone": ["LZ_WEST"],
            "Hub": [None],
        },
    )

    result = normalize_settlement_point_mapping(mapping)

    assert result.iloc[0]["reference_hub"] == "HB_WEST"
    assert result.iloc[0]["mapping_confidence"] == "Derived from load zone"


def test_location_leaderboards_and_explanations_render_with_missing_da():
    ts = pd.date_range("2026-04-24 00:00", periods=4, freq="15min", tz="America/Chicago")
    rt = pd.DataFrame(
        {
            "timestamp": list(ts) * 2,
            "settlement_point": ["HB_HOUSTON"] * 4 + ["RNODE_A"] * 4,
            "rt_price": [20, 20, 20, 20, 20, 40, 100, 160],
        },
    )

    result = build_location_intelligence(rt, pd.DataFrame(), volatility_window=3)
    high, low = price_spike_leaderboards(result)
    spreads = spread_leaderboard(result)
    volatility = volatility_leaderboard(result)
    explanations = top_location_explanations(result)

    assert high.iloc[0]["settlement_point"] == "RNODE_A"
    assert not low.empty
    assert spreads.empty
    assert not volatility.empty
    assert explanations
