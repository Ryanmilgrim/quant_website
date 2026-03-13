from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, flash, render_template, request

from website.lib.data import SUPPORTED_INDUSTRY_UNIVERSES
from website.web.services.universe_cache import (
    get_universe_returns_cached,
    get_universe_start_date_cached,
)

universe_bp = Blueprint("universe", __name__, url_prefix="/universe")


def _prepare_chart_payload(
    cumulative_growth: pd.DataFrame,
    *,
    rebase_to_100: bool = True,
    y_axis_title: str = "Benchmark index level (base = 100)",
) -> dict[str, object]:
    """Prepare a Plotly-friendly payload."""

    df = cumulative_growth.copy()

    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    if rebase_to_100:
        # Rebase so every chart starts at the same anchor and avoids runaway scales
        # when the cache includes prior history.
        df = df.div(df.iloc[0]).mul(100)

    dates = [dt.strftime("%Y-%m-%d") for dt in df.index]
    series = [
        {"name": col, "x": dates, "y": df[col].round(4).tolist()}
        for col in df.columns
    ]

    return {"series": series, "y_axis_title": y_axis_title}


@universe_bp.route("/", methods=["GET", "POST"])
@universe_bp.route("/historical", methods=["GET", "POST"])
def investment_universe():
    chart_data: Optional[dict] = None

    selected_universe = request.form.get("universe") or "5"
    weighting = "value"
    transform = request.form.get("transform", "index")
    frequency = request.form.get("frequency", "monthly")
    start_year_value = request.form.get("start_year")
    current_year = date.today().year
    earliest_start_year: Optional[int] = None
    start_year_display: Optional[str] = start_year_value

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if transform not in {"index", "log"}:
            raise ValueError("Unsupported transform option")

        if frequency not in {"daily", "monthly"}:
            raise ValueError("Unsupported frequency option")

        earliest_start = get_universe_start_date_cached(universe, weighting)
        earliest_start_year = earliest_start.year

        if not start_year_display:
            start_year_display = str(earliest_start_year)

        try:
            start_year = int(start_year_display)
        except ValueError as exc:
            raise ValueError("Invalid start year") from exc

        if start_year < earliest_start_year or start_year > current_year:
            raise ValueError("Start year out of range")

        start_date = date(start_year, 1, 1)

        df = get_universe_returns_cached(universe, weighting=weighting, start_date=start_date)
        asset_returns = df["assets"] if not df.empty else df

        if not df.empty:
            benchmark = df["benchmarks"]["Mkt"].rename("Benchmark")
            asset_returns = asset_returns.copy()
            asset_returns.insert(0, "Benchmark", benchmark)

        if frequency == "monthly":
            asset_returns = asset_returns.resample("M").sum()

        if asset_returns.empty:
            flash("No data returned for the requested range.", "warning")
        else:
            cumulative_log = asset_returns.cumsum()
            price_index = np.exp(cumulative_log) * 100

            if transform == "log":
                log_growth = cumulative_log.sub(cumulative_log.iloc[0])
                chart_data = _prepare_chart_payload(
                    log_growth,
                    rebase_to_100=False,
                    y_axis_title="Log price growth (relative to start)",
                )
            else:
                chart_data = _prepare_chart_payload(
                    price_index,
                )

            if chart_data is not None:
                chart_data["frequency"] = frequency
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to retrieve Fama-French industry data right now.", "danger")

    return render_template(
        "historical.html",
        chart_data=chart_data,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        transform=transform,
        frequency=frequency,
        start_year_value=start_year_display,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )
