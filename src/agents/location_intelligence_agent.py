from __future__ import annotations

import pandas as pd


LOCATION_COLUMNS = [
    "timestamp",
    "settlement_point",
    "settlement_point_display",
    "settlement_point_type",
    "reference_hub",
    "reference_hub_rt_price",
    "deviation_from_reference_hub",
    "mapping_confidence",
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


def load_zone_to_reference_hub(value: str | None) -> str | None:
    upper = str(value or "").upper()
    if "HOUSTON" in upper or upper in {"LZ_HOUSTON", "HOUSTON"}:
        return "HB_HOUSTON"
    if "NORTH" in upper or upper in {"LZ_NORTH", "NORTH"}:
        return "HB_NORTH"
    if "WEST" in upper or upper in {"LZ_WEST", "WEST"}:
        return "HB_WEST"
    if "SOUTH" in upper or upper in {"LZ_SOUTH", "SOUTH"}:
        return "HB_SOUTH"
    return None


def normalize_reference_hub(value: str | None) -> str | None:
    upper = str(value or "").upper().strip()
    if not upper or upper in {"NAN", "NONE"}:
        return None
    if upper.startswith("HB_"):
        return upper
    if "HOUSTON" in upper:
        return "HB_HOUSTON"
    if "NORTH" in upper:
        return "HB_NORTH"
    if "WEST" in upper:
        return "HB_WEST"
    if "SOUTH" in upper:
        return "HB_SOUTH"
    return None


def normalize_settlement_point_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["settlement_point", "settlement_load_zone", "reference_hub", "mapping_confidence"]
    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame(columns=columns)

    df = mapping_df.copy()
    resource_col = _first_existing(df, ["Resource Node", "RESOURCE_NODE", "resource_node"])
    zone_col = _first_existing(df, ["Settlement Load Zone", "SETTLEMENT_LOAD_ZONE", "settlement_load_zone"])
    hub_col = _first_existing(df, ["Hub", "HUB", "hub"])
    if resource_col is None:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame({"settlement_point": df[resource_col].astype(str).str.upper().str.strip()})
    out = out[out["settlement_point"].notna() & ~out["settlement_point"].isin(["", "NAN", "NONE"])]
    out["settlement_load_zone"] = df[zone_col].astype(str).str.upper().str.strip() if zone_col else pd.NA
    hub_values = df[hub_col] if hub_col else pd.Series([None] * len(df), index=df.index)
    official_hub = hub_values.map(normalize_reference_hub)
    derived_hub = out["settlement_load_zone"].map(load_zone_to_reference_hub)
    out["reference_hub"] = official_hub.fillna(derived_hub)
    out["mapping_confidence"] = "Unmapped fallback"
    out.loc[official_hub.notna(), "mapping_confidence"] = "Official mapping"
    out.loc[official_hub.isna() & derived_hub.notna(), "mapping_confidence"] = "Derived from load zone"
    return out.drop_duplicates("settlement_point")[columns].reset_index(drop=True)


def settlement_point_display(settlement_point: str, reference_hub: str | None) -> str:
    if reference_hub is None or pd.isna(reference_hub):
        return f"{settlement_point} · Unmapped"
    if settlement_point == reference_hub:
        return settlement_point
    return f"{settlement_point} · {reference_hub}"


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


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


def build_location_intelligence(rt_prices: pd.DataFrame, da_prices: pd.DataFrame, mapping_df: pd.DataFrame | None = None, volatility_window: int = 8) -> pd.DataFrame:
    if isinstance(mapping_df, int):
        volatility_window = mapping_df
        mapping_df = None
    if not isinstance(volatility_window, int) or volatility_window < 1:
        volatility_window = 8

    if rt_prices.empty or not {"timestamp", "settlement_point", "rt_price"}.issubset(rt_prices.columns):
        return _empty_location_frame()

    out = rt_prices[["timestamp", "settlement_point", "rt_price"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["settlement_point"] = out["settlement_point"].astype(str).str.upper().str.strip()
    out["settlement_point_type"] = out["settlement_point"].map(classify_settlement_point)

    if not da_prices.empty and {"timestamp", "settlement_point", "da_price"}.issubset(da_prices.columns):
        da = da_prices[["timestamp", "settlement_point", "da_price"]].copy()
        da["timestamp"] = pd.to_datetime(da["timestamp"]).dt.floor("h")
        da["settlement_point"] = da["settlement_point"].astype(str).str.upper().str.strip()
        da = da.sort_values("timestamp").drop_duplicates(["timestamp", "settlement_point"], keep="last")
        out["da_hour"] = out["timestamp"].dt.floor("h")
        out = out.merge(da, left_on=["da_hour", "settlement_point"], right_on=["timestamp", "settlement_point"], how="left", suffixes=("", "_da"))
        out = out.drop(columns=["da_hour", "timestamp_da"], errors="ignore")
    else:
        out["da_price"] = pd.NA

    out["da_price"] = pd.to_numeric(out["da_price"], errors="coerce")
    out["rt_da_spread"] = out["rt_price"] - out["da_price"]
    mapping = normalize_settlement_point_mapping(mapping_df if mapping_df is not None else pd.DataFrame())
    if not mapping.empty:
        out = out.merge(mapping, on="settlement_point", how="left")
    else:
        out["settlement_load_zone"] = pd.NA
        out["reference_hub"] = pd.NA
        out["mapping_confidence"] = "Unmapped fallback"
    hub_self = out["settlement_point"].where(out["settlement_point_type"] == "hub")
    zone_reference = out["settlement_point"].map(load_zone_to_reference_hub).where(out["settlement_point_type"] == "load_zone")
    out["reference_hub"] = out["reference_hub"].fillna(hub_self).fillna(zone_reference)
    out.loc[out["settlement_point_type"] == "hub", "mapping_confidence"] = "Self hub"
    out.loc[(out["settlement_point_type"] == "load_zone") & out["reference_hub"].notna(), "mapping_confidence"] = "Derived from load zone"
    out["mapping_confidence"] = out["mapping_confidence"].fillna("Unmapped fallback")

    hub_prices = (
        out[out["settlement_point_type"] == "hub"][["timestamp", "settlement_point", "rt_price"]]
        .rename(columns={"settlement_point": "reference_hub", "rt_price": "reference_hub_rt_price"})
        .drop_duplicates(["timestamp", "reference_hub"])
    )
    out = out.merge(hub_prices, on=["timestamp", "reference_hub"], how="left")
    out["deviation_from_reference_hub"] = out["rt_price"] - out["reference_hub_rt_price"]

    out["system_average_rt"] = out.groupby("timestamp")["rt_price"].transform("mean")
    type_average = out.groupby(["timestamp", "settlement_point_type"])["rt_price"].transform("mean")
    hub_average = out[out["settlement_point_type"] == "hub"].groupby("timestamp")["rt_price"].mean().rename("hub_average_rt")
    out = out.merge(hub_average, on="timestamp", how="left")
    out["reference_average_rt"] = type_average.where(out["settlement_point_type"].isin(["hub", "load_zone"]), out["hub_average_rt"])
    out["reference_average_rt"] = out["reference_average_rt"].fillna(out["system_average_rt"])
    out["deviation_from_reference"] = (out["deviation_from_reference_hub"]).fillna(out["rt_price"] - out["reference_average_rt"])
    out["deviation_from_hub_or_zone_average"] = out["deviation_from_reference"]
    out["deviation_from_system_average"] = out["rt_price"] - out["system_average_rt"]
    out["settlement_point_display"] = out.apply(lambda row: settlement_point_display(row["settlement_point"], row.get("reference_hub")), axis=1)

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
    cols = ["settlement_point_display", "settlement_point", "settlement_point_type", "reference_hub", "mapping_confidence", "rt_price", "deviation_from_system_average", "deviation_from_reference_hub", "congestion_label"]
    return (
        latest.nlargest(limit, "rt_price")[cols],
        latest.nsmallest(limit, "rt_price")[cols],
    )


def spread_leaderboard(location_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    latest = latest_location_snapshot(location_df).dropna(subset=["rt_da_spread"])
    if latest.empty:
        return latest
    cols = ["settlement_point_display", "settlement_point", "settlement_point_type", "reference_hub", "rt_price", "da_price", "rt_da_spread", "deviation_from_reference_hub", "congestion_label"]
    return latest.nlargest(limit, "rt_da_spread")[cols]


def volatility_leaderboard(location_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    latest = latest_location_snapshot(location_df).dropna(subset=["volatility_score"])
    if latest.empty:
        return latest
    cols = ["settlement_point_display", "settlement_point", "settlement_point_type", "reference_hub", "rt_price", "volatility_score", "volatility_label", "congestion_label"]
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
        reference_hub = row.get("reference_hub")
        if pd.notna(row.get("deviation_from_reference_hub")) and abs(row["deviation_from_reference_hub"]) >= 25 and pd.notna(reference_hub):
            drivers.append(f"pricing {row['deviation_from_reference_hub']:+.2f}/MWh versus {reference_hub}")
        elif pd.notna(row.get("deviation_from_reference")) and abs(row["deviation_from_reference"]) >= 25:
            drivers.append("divergence from its fallback reference average")
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
