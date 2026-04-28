from __future__ import annotations

import pandas as pd


LOCATION_COLUMNS = [
    "timestamp",
    "settlement_point",
    "settlement_point_type",
    "rt_price",
    "da_price",
    "rt_da_spread",
    "deviation_from_hub_or_zone_average",
    "deviation_from_reference",
    "volatility_score",
    "volatility_label",
    "congestion_score",
    "congestion_label",
]


def classify_settlement_point(name: str) -> str:
    upper = str(name).upper()
    if upper.startswith("HB_") or "HUB" in upper:
        return "hub"
    if upper.startswith("LZ_") or upper.startswith("DC_") or "LOAD_ZONE" in upper:
        return "load_zone"
    return "resource_node"


def volatility_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Stable"
    if value >= 75:
        return "Extreme"
    if value >= 25:
        return "Volatile"
    return "Stable"


def congestion_label(score: int) -> str:
    if score >= 5:
        return "High congestion"
    if score >= 2:
        return "Medium congestion"
    return "Low congestion"


def _empty_location_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LOCATION_COLUMNS)


def build_location_intelligence(rt_prices: pd.DataFrame, da_prices: pd.DataFrame, volatility_window: int = 8) -> pd.DataFrame:
    if rt_prices.empty or not {"timestamp", "settlement_point", "rt_price"}.issubset(rt_prices.columns):
        return _empty_location_frame()

    out = rt_prices[["timestamp", "settlement_point", "rt_price"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["settlement_point_type"] = out["settlement_point"].map(classify_settlement_point)

    if not da_prices.empty and {"timestamp", "settlement_point", "da_price"}.issubset(da_prices.columns):
        da = da_prices[["timestamp", "settlement_point", "da_price"]].copy()
        da["timestamp"] = pd.to_datetime(da["timestamp"]).dt.floor("h")
        da = da.sort_values("timestamp").drop_duplicates(["timestamp", "settlement_point"], keep="last")
        out["da_hour"] = out["timestamp"].dt.floor("h")
        out = out.merge(da, left_on=["da_hour", "settlement_point"], right_on=["timestamp", "settlement_point"], how="left", suffixes=("", "_da"))
        out = out.drop(columns=["da_hour", "timestamp_da"], errors="ignore")
    else:
        out["da_price"] = pd.NA

    out["da_price"] = pd.to_numeric(out["da_price"], errors="coerce")
    out["rt_da_spread"] = out["rt_price"] - out["da_price"]
    out["system_average_rt"] = out.groupby("timestamp")["rt_price"].transform("mean")
    type_average = out.groupby(["timestamp", "settlement_point_type"])["rt_price"].transform("mean")
    hub_average = out[out["settlement_point_type"] == "hub"].groupby("timestamp")["rt_price"].mean().rename("hub_average_rt")
    out = out.merge(hub_average, on="timestamp", how="left")
    out["reference_average_rt"] = type_average.where(out["settlement_point_type"].isin(["hub", "load_zone"]), out["hub_average_rt"])
    out["reference_average_rt"] = out["reference_average_rt"].fillna(out["system_average_rt"])
    out["deviation_from_reference"] = out["rt_price"] - out["reference_average_rt"]
    out["deviation_from_hub_or_zone_average"] = out["deviation_from_reference"]
    out["deviation_from_system_average"] = out["rt_price"] - out["system_average_rt"]

    out = out.sort_values(["settlement_point", "timestamp"])
    out["price_jump"] = out.groupby("settlement_point")["rt_price"].diff().abs()
    out["volatility_score"] = (
        out.groupby("settlement_point")["rt_price"]
        .rolling(window=volatility_window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    out["volatility_label"] = out["volatility_score"].map(volatility_label)

    # Congestion is a proxy, not an ERCOT constraint shadow price. It combines
    # local price separation, RT/DA divergence, and sudden interval-to-interval jumps.
    deviation_abs = out["deviation_from_reference"].abs()
    spread_abs = out["rt_da_spread"].abs().fillna(0)
    jump_abs = out["price_jump"].abs().fillna(0)
    out["congestion_score"] = 0
    out.loc[deviation_abs >= 25, "congestion_score"] += 1
    out.loc[deviation_abs >= 75, "congestion_score"] += 2
    out.loc[spread_abs >= 25, "congestion_score"] += 1
    out.loc[spread_abs >= 75, "congestion_score"] += 2
    out.loc[jump_abs >= 25, "congestion_score"] += 1
    out.loc[jump_abs >= 75, "congestion_score"] += 2
    out["congestion_label"] = out["congestion_score"].map(congestion_label)

    return out.sort_values("timestamp").reset_index(drop=True)


def latest_location_snapshot(location_df: pd.DataFrame) -> pd.DataFrame:
    if location_df.empty:
        return _empty_location_frame()
    return location_df.sort_values("timestamp").groupby("settlement_point", as_index=False).tail(1).reset_index(drop=True)


def price_spike_leaderboards(location_df: pd.DataFrame, limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = latest_location_snapshot(location_df)
    if latest.empty:
        return latest, latest
    cols = ["settlement_point", "settlement_point_type", "rt_price", "deviation_from_system_average", "deviation_from_hub_or_zone_average", "congestion_label"]
    return (
        latest.nlargest(limit, "rt_price")[cols],
        latest.nsmallest(limit, "rt_price")[cols],
    )


def spread_leaderboard(location_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    latest = latest_location_snapshot(location_df).dropna(subset=["rt_da_spread"])
    if latest.empty:
        return latest
    cols = ["settlement_point", "settlement_point_type", "rt_price", "da_price", "rt_da_spread", "congestion_label"]
    return latest.nlargest(limit, "rt_da_spread")[cols]


def volatility_leaderboard(location_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    latest = latest_location_snapshot(location_df).dropna(subset=["volatility_score"])
    if latest.empty:
        return latest
    cols = ["settlement_point", "settlement_point_type", "rt_price", "volatility_score", "volatility_label", "congestion_label"]
    return latest.nlargest(limit, "volatility_score")[cols]


def top_location_explanations(location_df: pd.DataFrame, limit: int = 3) -> list[str]:
    latest = latest_location_snapshot(location_df)
    if latest.empty:
        return ["Insufficient settlement point data to identify abnormal locations."]
    abnormal = latest.sort_values(
        ["congestion_score", "volatility_score", "deviation_from_reference"],
        ascending=[False, False, False],
    ).head(limit)
    explanations = []
    for _, row in abnormal.iterrows():
        drivers = []
        if pd.notna(row.get("deviation_from_reference")) and abs(row["deviation_from_reference"]) >= 25:
            drivers.append("divergence from the hub or zone reference average")
        if pd.notna(row.get("rt_da_spread")) and abs(row["rt_da_spread"]) >= 25:
            drivers.append("a widening RT versus DA spread")
        if pd.notna(row.get("price_jump")) and row["price_jump"] >= 25:
            drivers.append("a sudden real-time price move")
        if pd.notna(row.get("volatility_score")) and row["volatility_score"] >= 25:
            drivers.append("elevated rolling volatility")
        driver_text = ", ".join(drivers) if drivers else "limited but notable price separation"
        explanations.append(
            f"{row['settlement_point']} shows abnormal pricing due to {driver_text}. "
            "This may indicate localized congestion, outage effects, renewable variability, or settlement-point-specific imbalance."
        )
    return explanations
