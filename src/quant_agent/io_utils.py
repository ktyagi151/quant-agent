"""Cache paths and parquet helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def project_root() -> Path:
    """Return the repo root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def raw_dir() -> Path:
    p = data_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def meta_dir() -> Path:
    p = data_dir() / "meta"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = project_root() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ticker_path(ticker: str) -> Path:
    return raw_dir() / f"{ticker}.parquet"


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
