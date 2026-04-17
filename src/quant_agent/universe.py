"""S&P 500 universe — scraped from Wikipedia.

Two views:
  * `get_sp500_tickers()`       → current constituents (survivorship-biased)
  * `build_membership_matrix()` → point-in-time boolean date×ticker mask,
    reconstructed by unwinding Wikipedia's "Selected changes" table from today.

The changes-table reconstruction is only as good as Wikipedia: coverage is
robust back to ~1990s; earlier dates will over-include today's survivors.
"""
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .io_utils import meta_dir

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA = "Mozilla/5.0 (quant-agent research; +https://github.com/local)"


def _normalize_ticker(t: str) -> str:
    # yfinance uses dashes for class shares (BRK-B, BF-B) where Wikipedia uses dots.
    return t.strip().upper().replace(".", "-")


def get_sp500_tickers(refresh: bool = False) -> list[str]:
    """Return the current S&P 500 tickers, caching a dated snapshot."""
    if not refresh:
        snap = load_latest_snapshot()
        if snap is not None:
            return snap["ticker"].tolist()

    resp = requests.get(WIKI_URL, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df = df[["ticker", "name", "sector"]].copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)

    snap_path = meta_dir() / f"sp500_{date.today().isoformat()}.parquet"
    df.to_parquet(snap_path, index=False)
    return df["ticker"].tolist()


def load_latest_snapshot() -> pd.DataFrame | None:
    files = sorted(meta_dir().glob("sp500_2*.parquet"))  # dated snapshots only; excludes sp500_changes/sp500_membership
    if not files:
        return None
    return pd.read_parquet(files[-1])


def latest_snapshot_path() -> Path | None:
    files = sorted(meta_dir().glob("sp500_2*.parquet"))  # dated snapshots only; excludes sp500_changes/sp500_membership
    return files[-1] if files else None


# ----- Point-in-time membership ---------------------------------------------

_CHANGES_CACHE = meta_dir() / "sp500_changes.parquet"
_MEMBERSHIP_CACHE = meta_dir() / "sp500_membership.parquet"


def _fetch_wiki_tables() -> list[pd.DataFrame]:
    resp = requests.get(WIKI_URL, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def get_membership_changes(refresh: bool = False) -> pd.DataFrame:
    """Scrape Wikipedia's 'Selected changes' table.

    Returns columns: effective_date (Timestamp), added (str|NaN), removed (str|NaN).
    Rows are sorted by effective_date ascending.
    """
    if not refresh and _CHANGES_CACHE.exists():
        return pd.read_parquet(_CHANGES_CACHE)

    tables = _fetch_wiki_tables()
    raw = tables[1].copy()
    flat_cols = []
    for col in raw.columns:
        if isinstance(col, tuple):
            top, sub = col
            if top == sub:
                flat_cols.append(str(top).strip().lower().replace(" ", "_"))
            else:
                flat_cols.append(f"{str(top).strip().lower()}_{str(sub).strip().lower()}")
        else:
            flat_cols.append(str(col).strip().lower().replace(" ", "_"))
    raw.columns = flat_cols

    date_col = next((c for c in raw.columns if "effective" in c or c == "date"), None)
    added_col = next((c for c in raw.columns if "added" in c and "ticker" in c), None)
    removed_col = next((c for c in raw.columns if "removed" in c and "ticker" in c), None)
    if date_col is None or added_col is None or removed_col is None:
        raise RuntimeError(f"unexpected Wikipedia schema: {list(raw.columns)}")

    out = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(raw[date_col], errors="coerce"),
            "added": raw[added_col].map(_clean_ticker),
            "removed": raw[removed_col].map(_clean_ticker),
        }
    )
    out = out.dropna(subset=["effective_date"])
    # Drop rows where both added and removed are empty (noise).
    out = out[(out["added"].notna()) | (out["removed"].notna())]
    out = out.sort_values("effective_date").reset_index(drop=True)

    out.to_parquet(_CHANGES_CACHE, index=False)
    return out


def _clean_ticker(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    return s.upper().replace(".", "-")


def build_membership_matrix(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    refresh: bool = False,
    business_days_only: bool = True,
) -> pd.DataFrame:
    """Reconstruct point-in-time S&P 500 membership.

    Returns a DataFrame[date × ticker] of bool. True means the ticker was
    a constituent on that date. Cached to parquet once per (start, end) call
    that needs a non-empty reconstruction.

    Algorithm: walk Wikipedia's changes table backward from today.
      * state starts as today's constituent set.
      * for each change D (newest → oldest) with D > current_date:
          unwind: remove "added", add back "removed".
      * membership on date d = state after unwinding all changes with D > d.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if str(end).lower() != "today" else pd.Timestamp(pd.Timestamp.utcnow().date())

    if business_days_only:
        idx = pd.bdate_range(start_ts, end_ts)
    else:
        idx = pd.date_range(start_ts, end_ts)

    current = set(get_sp500_tickers(refresh=False))
    changes = get_membership_changes(refresh=refresh)

    # Walk changes descending; the current `state` reflects membership on
    # all dates ≥ the most recent change already processed.
    changes_desc = changes.sort_values("effective_date", ascending=False).reset_index(drop=True)

    all_tickers = set(current)
    for _, row in changes.iterrows():
        if pd.notna(row["added"]):
            all_tickers.add(row["added"])
        if pd.notna(row["removed"]):
            all_tickers.add(row["removed"])
    tickers_sorted = sorted(all_tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers_sorted)}

    # Initialize boolean matrix, then fill by walking dates newest → oldest.
    mat = np.zeros((len(idx), len(tickers_sorted)), dtype=bool)
    state = set(current)
    ci = 0  # cursor into changes_desc
    for row_i in range(len(idx) - 1, -1, -1):
        d = idx[row_i]
        # Unwind all changes strictly after date d that haven't been processed.
        while ci < len(changes_desc) and changes_desc.loc[ci, "effective_date"] > d:
            c = changes_desc.loc[ci]
            if pd.notna(c["added"]):
                state.discard(c["added"])
            if pd.notna(c["removed"]):
                state.add(c["removed"])
            ci += 1
        # Snapshot.
        for t in state:
            j = ticker_idx.get(t)
            if j is not None:
                mat[row_i, j] = True

    df = pd.DataFrame(mat, index=idx, columns=tickers_sorted)
    df.index.name = "date"
    return df


def load_membership_matrix() -> pd.DataFrame | None:
    if _MEMBERSHIP_CACHE.exists():
        return pd.read_parquet(_MEMBERSHIP_CACHE)
    return None


def cache_membership_matrix(df: pd.DataFrame) -> Path:
    df.to_parquet(_MEMBERSHIP_CACHE)
    return _MEMBERSHIP_CACHE
