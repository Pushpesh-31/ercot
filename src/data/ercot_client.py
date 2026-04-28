from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.cache import save_frame, save_last_success
from src.utils.time_utils import ERCOT_TZ, now_ercot

SETTLEMENT_POINTS = ["HB_HOUSTON", "HB_NORTH", "HB_WEST", "HB_SOUTH"]


class ErcotDataError(RuntimeError):
    pass


@dataclass
class ErcotBundle:
    rt_prices: pd.DataFrame
    da_prices: pd.DataFrame
    load: pd.DataFrame
    wind: pd.DataFrame
    solar: pd.DataFrame
    last_refresh: str


def _secret(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def load_credentials() -> tuple[str | None, str | None, str | None]:
    load_dotenv()
    return (
        os.getenv("ERCOT_USERNAME") or _secret("ERCOT_USERNAME"),
        os.getenv("ERCOT_PASSWORD") or _secret("ERCOT_PASSWORD"),
        os.getenv("ERCOT_SUBSCRIPTION_KEY") or _secret("ERCOT_SUBSCRIPTION_KEY"),
    )


def credential_help() -> str:
    return (
        "ERCOT API credentials are required. Set ERCOT_USERNAME, ERCOT_PASSWORD, "
        "and ERCOT_SUBSCRIPTION_KEY in .env for local use or Streamlit secrets for deployment. "
        "Register at https://apiexplorer.ercot.com/ and subscribe to the ERCOT Public API."
    )


class ErcotClient:
    """Thin adapter around ERCOT public data using gridstatus' ERCOT API helpers."""

    def __init__(self) -> None:
        username, password, subscription_key = load_credentials()
        if not all([username, password, subscription_key]):
            raise ErcotDataError(credential_help())
        try:
            from gridstatus.ercot_api.ercot_api import ErcotAPI
        except ImportError as exc:
            raise ErcotDataError("gridstatus is required for ERCOT API access. Install requirements.txt.") from exc

        self.api = ErcotAPI(
            username=username,
            password=password,
            public_subscription_key=subscription_key,
        )

    def fetch_bundle(self, settlement_point: str, hours: int) -> ErcotBundle:
        if settlement_point not in SETTLEMENT_POINTS:
            raise ErcotDataError(f"Unsupported settlement point: {settlement_point}")

        end = now_ercot()
        start = end - pd.Timedelta(hours=hours)
        fetch_start = start.floor("d")
        fetch_end = (end + pd.Timedelta(days=1)).floor("d")

        try:
            rt_raw = self.api.get_spp_real_time_15_min(fetch_start, end=fetch_end)
            da_raw = self.api.get_spp_day_ahead_hourly(fetch_start, end=fetch_end)
            load_raw = self.api.get_load_forecast_by_model(fetch_start, end=fetch_end)
            wind_raw = self.api.get_wind_actual_and_forecast_hourly(fetch_start, end=fetch_end)
            solar_raw = self.api.get_solar_actual_and_forecast_hourly(fetch_start, end=fetch_end)
        except requests.RequestException as exc:
            raise ErcotDataError(f"ERCOT API request failed. {credential_help()} Details: {exc}") from exc
        except Exception as exc:
            raise ErcotDataError(f"Unable to pull ERCOT data. {credential_help()} Details: {exc}") from exc

        bundle = ErcotBundle(
            rt_prices=self._prices(rt_raw, settlement_point, start, end, "NP6-905-CD"),
            da_prices=self._prices(da_raw, settlement_point, start, end, "NP4-190-CD"),
            load=self._load(load_raw, start, end),
            wind=self._renewable(wind_raw, start, end, "wind_mw"),
            solar=self._renewable(solar_raw, start, end, "solar_mw"),
            last_refresh=now_ercot().isoformat(),
        )
        self._save_success(bundle, settlement_point, hours)
        return bundle

    def fetch_settlement_point_prices(self, hours: int) -> dict[str, pd.DataFrame]:
        end = now_ercot()
        start = end - pd.Timedelta(hours=hours)
        fetch_start = start.floor("d")
        fetch_end = (end + pd.Timedelta(days=1)).floor("d")

        try:
            rt_raw = self.api.get_spp_real_time_15_min(fetch_start, end=fetch_end)
            da_raw = self.api.get_spp_day_ahead_hourly(fetch_start, end=fetch_end)
        except requests.RequestException as exc:
            raise ErcotDataError(f"ERCOT settlement point price request failed. {credential_help()} Details: {exc}") from exc
        except Exception as exc:
            raise ErcotDataError(f"Unable to pull ERCOT settlement point prices. {credential_help()} Details: {exc}") from exc

        return {
            "rt_prices": self._all_prices(rt_raw, start, end, "NP6-905-CD", "rt_price"),
            "da_prices": self._all_prices(da_raw, start, end, "NP4-190-CD", "da_price"),
        }

    @staticmethod
    def fetch_settlement_point_mapping() -> pd.DataFrame:
        try:
            from gridstatus import Ercot
        except ImportError as exc:
            raise ErcotDataError("gridstatus is required for ERCOT settlement point mapping.") from exc

        try:
            return Ercot().get_settlement_points_electrical_bus_mapping("latest")
        except Exception as exc:
            raise ErcotDataError(f"Unable to pull ERCOT settlement point mapping. Details: {exc}") from exc

    @staticmethod
    def _filter_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if df.empty or "timestamp" not in df.columns:
            return df
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        if out["timestamp"].dt.tz is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize(ERCOT_TZ, ambiguous="NaT", nonexistent="NaT")
        return out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)

    def _prices(self, raw: pd.DataFrame, settlement_point: str, start: pd.Timestamp, end: pd.Timestamp, source: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", "settlement_point", "price", "source_dataset"])
        df = raw.copy()
        time_col = self._first_existing(df, ["Interval Start", "Delivery Date", "deliveryDate"])
        loc_col = self._first_existing(df, ["Location", "Settlement Point", "settlementPoint", "SettlementPoint"])
        price_col = self._first_existing(df, ["Settlement Point Price", "SPP", "Price", "settlementPointPrice"])
        if not all([time_col, loc_col, price_col]):
            return pd.DataFrame(columns=["timestamp", "settlement_point", "price", "source_dataset"])
        df = df[df[loc_col].astype(str).str.upper() == settlement_point.upper()]
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df[time_col]),
                "settlement_point": df[loc_col].astype(str),
                "price": pd.to_numeric(df[price_col], errors="coerce"),
                "source_dataset": source,
            },
        )
        return self._filter_window(out.dropna(subset=["timestamp"]), start, end)

    def _all_prices(self, raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, source: str, price_name: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", "settlement_point", price_name, "source_dataset"])
        df = raw.copy()
        time_col = self._first_existing(df, ["Interval Start", "Delivery Date", "deliveryDate"])
        loc_col = self._first_existing(df, ["Location", "Settlement Point", "settlementPoint", "SettlementPoint"])
        price_col = self._first_existing(df, ["Settlement Point Price", "SPP", "Price", "settlementPointPrice"])
        if not all([time_col, loc_col, price_col]):
            return pd.DataFrame(columns=["timestamp", "settlement_point", price_name, "source_dataset"])
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df[time_col]),
                "settlement_point": df[loc_col].astype(str),
                price_name: pd.to_numeric(df[price_col], errors="coerce"),
                "source_dataset": source,
            },
        )
        return self._filter_window(out.dropna(subset=["timestamp"]), start, end)

    def _load(self, raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", "load_mw", "source_dataset"])
        df = raw.copy()
        time_col = self._first_existing(df, ["Interval Start", "Delivery Hour", "timestamp"])
        publish_col = self._first_existing(df, ["Publish Time"])
        model_col = self._first_existing(df, ["Model"])
        value_cols = [col for col in ["System Total", "systemTotal"] if col in df.columns]
        if not value_cols:
            weather_zone_cols = [
                "Coast",
                "East",
                "Far West",
                "North",
                "North Central",
                "South Central",
                "Southern",
                "West",
            ]
            value_cols = [col for col in weather_zone_cols if col in df.columns]
        if not time_col or not value_cols:
            return pd.DataFrame(columns=["timestamp", "load_mw", "source_dataset"])
        if model_col and "Total" in df[model_col].astype(str).unique():
            df = df[df[model_col].astype(str) == "Total"]
        if publish_col:
            latest_publish = df[publish_col].max()
            df = df[df[publish_col] == latest_publish]
        numeric = df[value_cols].apply(pd.to_numeric, errors="coerce")
        if len(value_cols) > 1:
            values = numeric.sum(axis=1, min_count=1)
        else:
            values = numeric[value_cols[0]]
        out = pd.DataFrame({"timestamp": pd.to_datetime(df[time_col]), "load_mw": values, "source_dataset": "NP3-565-CD load forecast by model/weather zone"})
        return self._filter_window(out.dropna(subset=["timestamp"]), start, end)

    def _renewable(self, raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, output_col: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["timestamp", output_col, "source_dataset"])
        df = raw.copy()
        time_col = self._first_existing(df, ["Interval Start", "Delivery Hour", "timestamp"])
        publish_col = self._first_existing(df, ["Publish Time"])
        preferred = ["GEN SYSTEM WIDE", "genSystemWide"]
        actual_cols = [col for col in preferred if col in df.columns]
        if not actual_cols:
            actual_cols = [col for col in df.columns if "actual" in col.lower() or str(col).upper().startswith("GEN ")]
        value_col = (actual_cols or [None])[0]
        if not time_col or not value_col:
            return pd.DataFrame(columns=["timestamp", output_col, "source_dataset"])
        if publish_col:
            latest_by_interval = df.groupby(time_col)[publish_col].transform("max")
            df = df[df[publish_col] == latest_by_interval]
        source = "NP4-732-CD wind actual and forecast" if output_col == "wind_mw" else "NP4-737-CD solar actual and forecast"
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df[time_col]),
                output_col: pd.to_numeric(df[value_col], errors="coerce"),
                "source_dataset": source,
            },
        )
        return self._filter_window(out.dropna(subset=["timestamp"]), start, end)

    @staticmethod
    def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
        for name in names:
            if name in df.columns:
                return name
        lower = {str(col).lower(): col for col in df.columns}
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    @staticmethod
    def _save_success(bundle: ErcotBundle, settlement_point: str, hours: int) -> None:
        save_frame(bundle.rt_prices, "rt_prices")
        save_frame(bundle.da_prices, "da_prices")
        save_frame(bundle.load, "load")
        save_frame(bundle.wind, "wind")
        save_frame(bundle.solar, "solar")
        save_last_success({"last_refresh": bundle.last_refresh, "settlement_point": settlement_point, "hours": hours})


def bundle_to_dict(bundle: ErcotBundle) -> dict[str, Any]:
    return {
        "rt_prices": bundle.rt_prices,
        "da_prices": bundle.da_prices,
        "load": bundle.load,
        "wind": bundle.wind,
        "solar": bundle.solar,
        "last_refresh": bundle.last_refresh,
    }
