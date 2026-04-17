"""Yahoo Finance daily OHLCV fetch with parquet cache + incremental refresh."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .io_utils import raw_dir, read_parquet, ticker_path, write_parquet

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
BATCH_SIZE = 50


def _parse_date(d: str | datetime | pd.Timestamp) -> pd.Timestamp:
    if isinstance(d, str) and d.lower() == "today":
        return pd.Timestamp(datetime.utcnow().date())
    return pd.Timestamp(d)


def _normalize_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a single-ticker yfinance frame to our column schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    # yfinance sometimes omits "Adj Close" when auto_adjust=True; fallback.
    if "adj_close" not in out.columns:
        out["adj_close"] = out.get("close")
    out = out[[c for c in COLUMNS if c in out.columns]].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    return out


def _download_batch(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Batch-download via yfinance. Returns per-ticker normalized frames."""
    if not tickers:
        return {}
    raw = yf.download(
        tickers=" ".join(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return {t: pd.DataFrame(columns=COLUMNS) for t in tickers}
    if len(tickers) == 1:
        out[tickers[0]] = _normalize_yf_frame(raw)
        return out
    # MultiIndex columns: level 0 ticker, level 1 field.
    for t in tickers:
        if t in raw.columns.get_level_values(0):
            sub = raw[t].copy()
            out[t] = _normalize_yf_frame(sub)
        else:
            out[t] = pd.DataFrame(columns=COLUMNS)
    return out


def fetch_ohlcv(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    refresh: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV with incremental parquet cache.

    For each ticker, read existing cache, fetch only the missing tail
    (last_date+1 .. end), append, rewrite. When refresh=False, only reads cache.
    """
    tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order
    start_ts = _parse_date(start)
    end_ts = _parse_date(end)

    # Decide fetch window per ticker.
    to_fetch: dict[str, pd.Timestamp] = {}
    cached: dict[str, pd.DataFrame] = {}
    for t in tickers:
        p = ticker_path(t)
        if p.exists():
            df = read_parquet(p)
            cached[t] = df
            if refresh:
                last = df.index.max()
                if pd.isna(last) or last < end_ts:
                    fetch_from = (last + pd.Timedelta(days=1)) if pd.notna(last) else start_ts
                    to_fetch[t] = max(fetch_from, start_ts)
        else:
            if refresh:
                to_fetch[t] = start_ts
            else:
                cached[t] = pd.DataFrame(columns=COLUMNS)

    # Group by common start date for batching efficiency; simplest: one common start per batch.
    if to_fetch:
        items = list(to_fetch.items())
        for i in tqdm(range(0, len(items), BATCH_SIZE), desc="yfinance batches"):
            chunk = items[i : i + BATCH_SIZE]
            # Use the earliest fetch_from in the chunk; we'll slice per ticker after.
            chunk_start = min(s for _, s in chunk)
            chunk_tickers = [t for t, _ in chunk]
            new_frames = _download_batch(chunk_tickers, chunk_start, end_ts)
            for t, s in chunk:
                new_df = new_frames.get(t, pd.DataFrame(columns=COLUMNS))
                if not new_df.empty:
                    new_df = new_df.loc[new_df.index >= s]
                old = cached.get(t)
                if old is not None and not old.empty:
                    combined = pd.concat([old, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                else:
                    combined = new_df.sort_index()
                if not combined.empty:
                    write_parquet(combined, ticker_path(t))
                cached[t] = combined

    # Slice to requested window.
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = cached.get(t, pd.DataFrame(columns=COLUMNS))
        if not df.empty:
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        out[t] = df
    return out


def load_panel(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Read-only cache load (no network)."""
    return fetch_ohlcv(tickers, start, end, refresh=False)


def to_wide_panel(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convert per-ticker frames into {field: DataFrame[date x ticker]}."""
    fields = COLUMNS
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        cols = {}
        for t, df in raw.items():
            if df is None or df.empty or field not in df.columns:
                continue
            cols[t] = df[field]
        if cols:
            wide = pd.DataFrame(cols).sort_index()
            out[field] = wide
        else:
            out[field] = pd.DataFrame()
    return out
