from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path("data/cache")
LAST_SUCCESS_FILE = CACHE_DIR / "last_success.json"


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_frame(df: pd.DataFrame, name: str) -> Path:
    ensure_cache_dir()
    path = CACHE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_last_success(metadata: dict[str, Any]) -> None:
    ensure_cache_dir()
    LAST_SUCCESS_FILE.write_text(json.dumps(metadata, indent=2, default=str))


def load_last_success() -> dict[str, Any] | None:
    if not LAST_SUCCESS_FILE.exists():
        return None
    return json.loads(LAST_SUCCESS_FILE.read_text())
