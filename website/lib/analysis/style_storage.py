from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import pickle
import re
from typing import Optional

import pandas as pd

from .benchmark_style import StyleRun

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_snapshot_name(name: str) -> str:
    """Normalize a user-provided name into a safe, filesystem-friendly key."""
    if name is None:
        raise ValueError("Snapshot name is required.")
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    if not cleaned:
        raise ValueError("Snapshot name must include letters or numbers.")
    return cleaned[:80]


@dataclass
class StyleAnalysisSnapshot:
    """Persisted benchmark style analysis snapshot."""

    name: str
    created_at: datetime
    universe: int
    weighting: str
    factor_set: str
    start_date: Optional[date]
    end_date: Optional[date]
    run: StyleRun
    universe_data: pd.DataFrame

    @property
    def key(self) -> str:
        return normalize_snapshot_name(self.name)


@dataclass(frozen=True)
class StyleSnapshotInfo:
    """Lightweight metadata for listing saved snapshots."""

    key: str
    name: str
    created_at: datetime
    universe: int
    weighting: str
    factor_set: str
    start_date: Optional[date]
    end_date: Optional[date]
    window: Optional[int]
    window_years: Optional[float]
    window_frequency: Optional[str]
    rebalance: Optional[str]
    method: Optional[str]
    assets: int
    rebalance_start: Optional[date]
    rebalance_end: Optional[date]


def snapshot_path(directory: Path, name: str) -> Path:
    safe_name = normalize_snapshot_name(name)
    if safe_name.lower().endswith(".pkl"):
        safe_name = safe_name[:-4]
    return directory / f"{safe_name}.pkl"


def save_style_snapshot(
    snapshot: StyleAnalysisSnapshot,
    directory: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a style snapshot to a single pickle file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(directory, snapshot.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Snapshot already exists: {snapshot.name}")
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_style_snapshot(path: Path) -> StyleAnalysisSnapshot:
    """Load a style snapshot from a pickle file."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, StyleAnalysisSnapshot):
        raise TypeError("Invalid snapshot payload.")
    return obj


def list_style_snapshots(directory: Path) -> list[StyleSnapshotInfo]:
    """List saved snapshots in a directory."""
    if not directory.exists():
        return []
    infos: list[StyleSnapshotInfo] = []
    for path in sorted(directory.glob("*.pkl")):
        try:
            snapshot = load_style_snapshot(path)
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            continue
        infos.append(
            StyleSnapshotInfo(
                key=path.stem,
                name=snapshot.name,
                created_at=snapshot.created_at,
                universe=snapshot.universe,
                weighting=snapshot.weighting,
                factor_set=snapshot.factor_set,
                start_date=snapshot.start_date,
                end_date=snapshot.end_date,
                window=snapshot.run.params.get("style_window"),
                window_years=snapshot.run.params.get("style_window_years"),
                window_frequency=snapshot.run.params.get("window_frequency"),
                rebalance=snapshot.run.params.get("optimize_frequency"),
                method=snapshot.run.params.get("method"),
                assets=int(snapshot.run.weights.shape[1]) if not snapshot.run.weights.empty else 0,
                rebalance_start=(
                    snapshot.run.weights.index.min().date()
                    if not snapshot.run.weights.empty
                    else None
                ),
                rebalance_end=(
                    snapshot.run.weights.index.max().date()
                    if not snapshot.run.weights.empty
                    else None
                ),
            )
        )
    infos.sort(key=lambda item: item.created_at, reverse=True)
    return infos


__all__ = [
    "StyleAnalysisSnapshot",
    "StyleSnapshotInfo",
    "list_style_snapshots",
    "load_style_snapshot",
    "normalize_snapshot_name",
    "save_style_snapshot",
    "snapshot_path",
]
