from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ERCOT_TZ = "America/Chicago"


def now_ercot() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ERCOT_TZ)


def window_start(hours: int) -> pd.Timestamp:
    return now_ercot() - pd.Timedelta(hours=hours)


def iso_ts(value) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ERCOT_TZ)
    return ts.isoformat()


def briefing_filename(prefix: str = "ercot_briefing") -> str:
    stamp = datetime.now(ZoneInfo(ERCOT_TZ)).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.md"
