from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from website.lib.data.french_industry import (
    FactorSet,
    ReturnForm,
    Weighting,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)
from website.lib.returns import to_log_returns


def get_universe_returns(
    universe: int,
    *,
    weighting: Weighting = "value",
    factor_set: FactorSet = "ff3",
    return_form: ReturnForm = "log",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return daily Fama-French universe returns with factor and benchmark series.

    The result has MultiIndex columns with top-level groups:
    - "assets": industry portfolio returns for the selected universe
    - "factors": the selected factor set (minus Rf)
    - "benchmarks": Mkt (Mkt-Rf + Rf) and Rf

    Returns are expressed in decimal form. If ``return_form`` is "log", Mkt is
    computed from simple returns before the log transform is applied. Series
    are aligned on shared dates (inner join). Date filters are inclusive of
    ``start_date`` and exclusive of ``end_date``.
    """

    industries = fetch_ff_industry_daily(
        universe,
        weighting=weighting,
        return_form="simple",
        start_date=start_date,
        end_date=end_date,
    )
    factors = fetch_ff_factors_daily(
        factor_set=factor_set,
        return_form="simple",
        start_date=start_date,
        end_date=end_date,
    )

    benchmarks = pd.DataFrame(
        {
            "Mkt": factors["Mkt-Rf"] + factors["Rf"],
            "Rf": factors["Rf"],
        },
        index=factors.index,
    )

    combined = pd.concat(
        {
            "assets": industries,
            "factors": factors[[col for col in factors.columns if col != "Rf"]],
            "benchmarks": benchmarks,
        },
        axis=1,
        join="inner",
    )
    combined.columns.names = ["group", "series"]

    if return_form == "log":
        combined = to_log_returns(combined)
    elif return_form != "simple":
        raise ValueError("return_form must be 'simple' or 'log'.")

    if start_date:
        combined = combined.loc[combined.index >= pd.Timestamp(start_date)]
    if end_date:
        combined = combined.loc[combined.index < pd.Timestamp(end_date)]

    return combined


def get_universe_start_date(
    universe: int,
    weighting: Weighting,
    *,
    factor_set: FactorSet = "ff3",
) -> date:
    """Return the earliest available date for a universe/weighting/factor set."""
    df = get_universe_returns(
        universe,
        weighting=weighting,
        factor_set=factor_set,
        return_form="simple",
    )

    if df.empty:
        raise ValueError("No data available for the requested universe.")

    return df.index.min().date()
