from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd

from website.lib.data.french_industry import FactorSet, Weighting
from website.lib.universe import get_universe_returns


@lru_cache(maxsize=32)
def _get_full_universe(universe: int, weighting: Weighting, factor_set: FactorSet) -> pd.DataFrame:
    """Cache the full universe return panel for reuse across requests."""
    return get_universe_returns(
        universe,
        weighting=weighting,
        factor_set=factor_set,
        return_form="log",
    )


def get_universe_returns_cached(
    universe: int,
    *,
    weighting: Weighting,
    factor_set: FactorSet = "ff3",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return a filtered slice of cached universe returns."""
    df = _get_full_universe(universe, weighting, factor_set).copy()

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    return df


def get_universe_start_date_cached(
    universe: int,
    weighting: Weighting,
    *,
    factor_set: FactorSet = "ff3",
) -> date:
    """Return the earliest available date for a universe/weighting/factor set."""
    df = _get_full_universe(universe, weighting, factor_set)

    if df.empty:
        raise ValueError("No data available for the requested universe.")

    return df.index.min().date()


def clear_universe_cache() -> None:
    """Expose cache clearing for manual refreshes."""
    _get_full_universe.cache_clear()
