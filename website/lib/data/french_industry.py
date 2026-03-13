from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests

from website.lib.returns import to_log_returns

Weighting = Literal["value", "equal"]
JoinHow = Literal["inner", "outer"]
ReturnForm = Literal["simple", "log"]
FactorSet = Literal["ff3", "ff5", "ff3_mom", "ff5_mom"]

FRENCH_FTP_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Industry universes offered by the Data Library (daily versions exist for these)
SUPPORTED_INDUSTRY_UNIVERSES = (5, 10, 12, 17, 30, 38, 48, 49)
SUPPORTED_FACTOR_SETS = ("ff3", "ff5", "ff3_mom", "ff5_mom")

_INDUSTRY_ZIP = {n: f"{n}_Industry_Portfolios_daily_CSV.zip" for n in SUPPORTED_INDUSTRY_UNIVERSES}
_FACTORS_DAILY_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
_FACTORS_5_DAILY_ZIP = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOMENTUM_DAILY_ZIP = "F-F_Momentum_Factor_daily_CSV.zip"

_BASE_FACTOR_ZIPS = {
    "ff3": _FACTORS_DAILY_ZIP,
    "ff5": _FACTORS_5_DAILY_ZIP,
}

_BASE_FACTOR_COLUMNS = {
    "ff3": ("Mkt-Rf", "SMB", "HML", "Rf"),
    "ff5": ("Mkt-Rf", "SMB", "HML", "RMW", "CMA", "Rf"),
}

_SECTION_MARKER: dict[Weighting, str] = {
    "value": "Average Value Weighted Returns -- Daily",
    "equal": "Average Equal Weighted Returns -- Daily",
}

_MISSING_CODES = (-99.99, -999)


@dataclass(frozen=True)
class FrenchDownloadConfig:
    cache_dir: Path = Path.home() / ".cache" / "ken_french"
    timeout_s: float = 30.0
    max_retries: int = 4
    user_agent: str = "Mozilla/5.0 (compatible; quant_website/1.0)"


def _download_with_cache(url: str, dest: Path, cfg: FrenchDownloadConfig, refresh: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return dest

    headers = {"User-Agent": cfg.user_agent}
    last_exc: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=cfg.timeout_s) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return dest
        except Exception as e:
            last_exc = e
            if attempt < cfg.max_retries:
                time.sleep(2 ** (attempt - 1))
            else:
                raise RuntimeError(f"Failed to download {url} after {cfg.max_retries} attempts") from last_exc

    return dest  # unreachable


def _read_single_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.lower().endswith(".csv")), names[0])
        return zf.read(csv_name).decode("utf-8", errors="ignore")


def _normalize_factor_key(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if key == "umd":
        return "mom"
    return key


def _extract_sectioned_daily_table(csv_text: str, weighting: Weighting) -> str:
    lines = csv_text.splitlines()
    marker = _SECTION_MARKER[weighting]

    try:
        marker_idx = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as e:
        raise ValueError(f"Could not find section marker {marker!r} in file.") from e

    header_idx = marker_idx + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    if header_idx >= len(lines):
        raise ValueError(f"Found marker {marker!r} but no header row after it.")

    out_lines = [lines[header_idx].strip()]

    for raw in lines[header_idx + 1 :]:
        s = raw.strip()
        if not s:
            break
        if len(s) < 8 or not s[:8].isdigit():
            break
        out_lines.append(s)

    return "\n".join(out_lines)


def _extract_daily_factor_table(csv_text: str, required_keys: set[str]) -> str:
    lines = csv_text.splitlines()
    header_idx: Optional[int] = None

    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        fields = [_normalize_factor_key(f) for f in raw.split(",")]
        if required_keys.issubset(fields):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find the daily factor header row in the Fama-French file.")

    out_lines = [lines[header_idx].strip()]

    for raw in lines[header_idx + 1 :]:
        s = raw.strip()
        if not s:
            break
        if len(s) < 8 or not s[:8].isdigit():
            break
        out_lines.append(s)

    return "\n".join(out_lines)


def _clean_percent_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace({code: np.nan for code in _MISSING_CODES})
    return out / 100.0


def _normalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}

    for col in df.columns:
        key = _normalize_factor_key(col)
        if key == "mktrf":
            rename[col] = "Mkt-Rf"
        elif key == "smb":
            rename[col] = "SMB"
        elif key == "hml":
            rename[col] = "HML"
        elif key == "rmw":
            rename[col] = "RMW"
        elif key == "cma":
            rename[col] = "CMA"
        elif key == "mom":
            rename[col] = "Mom"
        elif key == "rf":
            rename[col] = "Rf"

    return df.rename(columns=rename)


def fetch_ff_industry_daily(
    universe: int,
    *,
    weighting: Weighting = "value",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    return_form: ReturnForm = "log",
    refresh: bool = False,
    cfg: FrenchDownloadConfig = FrenchDownloadConfig(),
) -> pd.DataFrame:
    """Fetch daily industry returns from the Fama-French Data Library."""
    if universe not in _INDUSTRY_ZIP:
        raise ValueError(f"Unsupported industry universe {universe}. Supported: {SUPPORTED_INDUSTRY_UNIVERSES}")

    zip_name = _INDUSTRY_ZIP[universe]
    url = f"{FRENCH_FTP_BASE}/{zip_name}"
    zip_path = _download_with_cache(url, cfg.cache_dir / zip_name, cfg, refresh)

    csv_text = _read_single_csv_from_zip(zip_path)
    table_text = _extract_sectioned_daily_table(csv_text, weighting)

    df = pd.read_csv(io.StringIO(table_text), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df.index.name = "Date"
    df = df.loc[df.index.notna()].sort_index()
    df = _clean_percent_returns(df)

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    if return_form == "log":
        df = to_log_returns(df)
    elif return_form != "simple":
        raise ValueError("return_form must be 'simple' or 'log'.")

    return df


def fetch_ff_factors_daily(
    *,
    factor_set: FactorSet = "ff3",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    return_form: ReturnForm = "log",
    refresh: bool = False,
    cfg: FrenchDownloadConfig = FrenchDownloadConfig(),
) -> pd.DataFrame:
    """Fetch daily Fama-French factor returns for the requested factor set.

    Supported factor sets:
    - ff3: Mkt-Rf, SMB, HML, Rf
    - ff5: Mkt-Rf, SMB, HML, RMW, CMA, Rf
    - ff3_mom / ff5_mom: add momentum (Mom) from the separate momentum file
    """
    if factor_set not in SUPPORTED_FACTOR_SETS:
        raise ValueError(f"Unsupported factor_set {factor_set!r}. Supported: {SUPPORTED_FACTOR_SETS}")

    include_mom = factor_set.endswith("_mom")
    base_set = factor_set.replace("_mom", "")
    base_zip = _BASE_FACTOR_ZIPS.get(base_set)
    base_cols = _BASE_FACTOR_COLUMNS.get(base_set)

    if base_zip is None or base_cols is None:
        raise ValueError(f"Unsupported factor_set {factor_set!r}. Supported: {SUPPORTED_FACTOR_SETS}")

    base_required = {_normalize_factor_key(c) for c in base_cols}
    url = f"{FRENCH_FTP_BASE}/{base_zip}"
    zip_path = _download_with_cache(url, cfg.cache_dir / base_zip, cfg, refresh)

    csv_text = _read_single_csv_from_zip(zip_path)
    table_text = _extract_daily_factor_table(csv_text, base_required)

    df = pd.read_csv(io.StringIO(table_text), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df.index.name = "Date"
    df = df.loc[df.index.notna()].sort_index()
    df = _clean_percent_returns(df)
    df = _normalize_factor_columns(df)

    required = set(base_cols)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected factor columns: {sorted(missing)}")

    df = df[list(base_cols)]

    if include_mom:
        mom_required = {_normalize_factor_key("Mom")}
        mom_url = f"{FRENCH_FTP_BASE}/{_MOMENTUM_DAILY_ZIP}"
        mom_zip = _download_with_cache(mom_url, cfg.cache_dir / _MOMENTUM_DAILY_ZIP, cfg, refresh)
        mom_csv = _read_single_csv_from_zip(mom_zip)
        mom_table = _extract_daily_factor_table(mom_csv, mom_required)
        mom_df = pd.read_csv(io.StringIO(mom_table), index_col=0)
        mom_df.index = pd.to_datetime(mom_df.index.astype(str), format="%Y%m%d", errors="coerce")
        mom_df.index.name = "Date"
        mom_df = mom_df.loc[mom_df.index.notna()].sort_index()
        mom_df = _clean_percent_returns(mom_df)
        mom_df = _normalize_factor_columns(mom_df)

        if "Mom" not in mom_df.columns:
            raise ValueError("Missing expected factor column: Mom")

        mom_df = mom_df[["Mom"]]
        df = df.join(mom_df, how="inner")
        df = df[list(base_cols[:-1]) + ["Mom"] + [base_cols[-1]]]

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    if return_form == "log":
        df = to_log_returns(df)
    elif return_form != "simple":
        raise ValueError("return_form must be 'simple' or 'log'.")

    return df


__all__ = [
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "SUPPORTED_FACTOR_SETS",
    "FactorSet",
    "fetch_ff_industry_daily",
    "fetch_ff_factors_daily",
]
