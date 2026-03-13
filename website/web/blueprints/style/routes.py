from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from website.lib.analysis import (
    StyleAnalysis,
    StyleAnalysisSnapshot,
    StyleRun,
    list_style_snapshots,
    load_style_snapshot,
    save_style_snapshot,
    snapshot_path,
)
from website.lib.data import SUPPORTED_INDUSTRY_UNIVERSES
from website.web.services.universe_cache import (
    get_universe_returns_cached,
    get_universe_start_date_cached,
)

style_bp = Blueprint("style", __name__, url_prefix="/style")

_STEPS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def _style_results_dir() -> Path:
    root = Path(current_app.instance_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_results" / "benchmark_style"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _opt_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null", "auto"):
        return None
    return int(s)


def _opt_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null", "auto"):
        return None
    return float(s)


def _top_weights(weights: pd.Series, *, top_n: int = 10) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    w = weights.astype(float).clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s
    ranked = w.sort_values(ascending=False)
    top = ranked.head(top_n).copy()
    other = float(ranked.iloc[top_n:].sum())
    if other > 0:
        top.loc["Other"] = other
    return top[top > 0]


def _prepare_line_chart_payload(
    df: pd.DataFrame,
    *,
    y_axis_title: str,
    round_to: int = 6,
) -> dict[str, object]:
    """Prepare a Plotly-friendly payload."""
    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    data = df.copy()

    dates = [dt.strftime("%Y-%m-%d") for dt in data.index]
    series = [
        {"name": col, "x": dates, "y": data[col].round(round_to).tolist()}
        for col in data.columns
    ]
    return {"series": series, "y_axis_title": y_axis_title}


def _prepare_weights_history_payload(
    weights: pd.DataFrame,
) -> dict[str, object]:
    if weights.empty:
        return {"series": [], "y_axis_title": "Weight (%)"}

    data = weights.copy().astype(float).clip(lower=0.0)
    row_sum = data.sum(axis=1).replace(0.0, np.nan)
    data = data.div(row_sum, axis=0).fillna(0.0)

    data = data.mul(100.0)
    dates = [dt.strftime("%Y-%m-%d") for dt in data.index]
    series = [
        {"name": col, "x": dates, "y": data[col].round(2).tolist()}
        for col in data.columns
    ]
    return {"series": series, "y_axis_title": "Weight (%)"}


def _perf_row(
    name: str,
    returns: pd.Series,
    *,
    steps_per_year: int,
    info_ratio: Optional[float] = None,
) -> dict[str, object]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {
            "name": name,
            "total_cont_return": None,
            "ann_cont_return": None,
            "ann_vol": None,
            "sharpe": None,
            "info_ratio": info_ratio,
        }

    total_cont_return = float(r.sum() * 100.0)
    ann_cont_return = float(r.mean() * steps_per_year * 100.0)

    vol = float(r.std())
    ann_vol = float(vol * np.sqrt(steps_per_year) * 100.0) if np.isfinite(vol) else None

    if np.isfinite(vol) and vol > 0:
        sharpe = float((r.mean() / vol) * np.sqrt(steps_per_year))
    else:
        sharpe = None

    return {
        "name": name,
        "total_cont_return": total_cont_return,
        "ann_cont_return": ann_cont_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "info_ratio": info_ratio,
    }


def _summarize_style_run(run: StyleRun) -> dict[str, object]:
    chart_growth: Optional[dict[str, object]] = None
    chart_tracking: Optional[dict[str, object]] = None
    weights_history: Optional[dict[str, object]] = None
    weights_table: list[dict[str, object]] = []
    metrics: Optional[dict[str, object]] = None
    warnings: list[str] = []

    weights = run.weights
    if weights.empty:
        warnings.append("No feasible weights found for the requested configuration.")
    else:
        min_weight = float(weights.min().min())
        if min_weight < -1e-8:
            warnings.append("Some weights are negative beyond tolerance.")

        row_sum = weights.sum(axis=1)
        if ((row_sum - 1.0).abs() > 1e-5).any():
            warnings.append("Weights do not sum to 100% on every rebalance date.")

        weights_history = _prepare_weights_history_payload(weights)

        top = _top_weights(weights.iloc[-1], top_n=10)
        if top.empty:
            warnings.append("No usable weights found for the latest rebalance date.")
        else:
            weights_table = [
                {
                    "name": str(name),
                    "weight": float(weight),
                    "weight_pct": float(weight * 100.0),
                }
                for name, weight in top.items()
            ]

    portfolio = run.portfolio_return.rename("Simulated Portfolio")
    benchmark = run.benchmark_return.rename("Market Benchmark")
    growth = pd.concat([portfolio, benchmark], axis=1).dropna(how="any")
    if not growth.empty:
        growth = np.exp(growth.cumsum()) * 100
        chart_growth = _prepare_line_chart_payload(
            growth,
            y_axis_title="Indexed growth (base = 100)",
        )

    active_return = run.tracking_error.rename("Active Return").dropna()
    if not active_return.empty:
        active_return_pct = active_return.mul(100.0).rename("Residual return")
        chart_tracking = _prepare_line_chart_payload(
            active_return_pct.to_frame(),
            y_axis_title="Daily return",
            round_to=4,
        )
        control_sd = float(active_return_pct.std())
        if np.isfinite(control_sd) and control_sd > 0:
            chart_tracking["control_limits"] = {
                "upper": float(3.0 * control_sd),
                "lower": float(-3.0 * control_sd),
            }

    te = run.tracking_error.dropna()
    window_frequency = str(run.params.get("window_frequency") or "daily")
    steps_per_year = _STEPS_PER_YEAR.get(window_frequency, 252)

    info_ratio = None
    if not te.empty:
        te_vol = float(te.std())
        if np.isfinite(te_vol) and te_vol > 0:
            info_ratio = float((float(te.mean()) / te_vol) * np.sqrt(steps_per_year))

    sample = pd.concat([portfolio, benchmark], axis=1).dropna(how="any")
    sample_start = sample.index.min() if not sample.empty else None
    sample_end = sample.index.max() if not sample.empty else None
    sample_years = (
        float((sample_end - sample_start).days / 365.25)
        if sample_start is not None and sample_end is not None
        else None
    )

    metrics = {
        "inputs": {
            "window": run.params.get("style_window"),
            "window_years": run.params.get("style_window_years"),
            "window_frequency": window_frequency,
            "rebalance": run.params.get("optimize_frequency"),
            "method": run.params.get("method"),
            "assets": weights.shape[1] if not weights.empty else 0,
            "rebalance_start": (
                weights.index.min().strftime("%Y-%m-%d") if not weights.empty else None
            ),
            "rebalance_end": (
                weights.index.max().strftime("%Y-%m-%d") if not weights.empty else None
            ),
        },
        "sample": {
            "start": sample_start.strftime("%Y-%m-%d") if sample_start is not None else None,
            "end": sample_end.strftime("%Y-%m-%d") if sample_end is not None else None,
            "years": sample_years,
        },
        "performance": [
            _perf_row(
                "Simulated Portfolio",
                portfolio,
                steps_per_year=steps_per_year,
            ),
            _perf_row(
                "Market Benchmark",
                benchmark,
                steps_per_year=steps_per_year,
            ),
            _perf_row(
                "Active Return",
                te,
                steps_per_year=steps_per_year,
                info_ratio=info_ratio,
            ),
        ],
    }

    return {
        "chart_growth": chart_growth,
        "chart_tracking": chart_tracking,
        "weights_history": weights_history,
        "weights_table": weights_table,
        "metrics": metrics,
        "warnings": warnings,
    }


@style_bp.route("/benchmark-style", methods=["GET", "POST"])
def benchmark_style():
    chart_growth: Optional[dict[str, object]] = None
    chart_tracking: Optional[dict[str, object]] = None
    weights_history: Optional[dict[str, object]] = None
    weights_table: list[dict[str, object]] = []
    metrics: Optional[dict[str, object]] = None

    params = request.values
    selected_universe = (params.get("universe") or "10").strip()
    style_window_value = (params.get("style_window") or "").strip()
    style_window_years_value = (params.get("style_window_years") or "").strip()
    rebalance = (params.get("rebalance") or "annual").strip()
    method = (params.get("method") or "projection").strip()
    start_year_value = (params.get("start_year") or "").strip()
    save_name_value = (params.get("save_name") or "").strip()
    save_overwrite = bool(params.get("save_overwrite"))

    weighting = "value"
    factor_set = "ff3"
    current_year = date.today().year
    earliest_start_year: Optional[int] = None
    start_year_display: Optional[str] = start_year_value

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if method not in {"projection", "qp"}:
            raise ValueError("Unsupported solve method")

        if rebalance not in {"daily", "weekly", "monthly", "annual"}:
            raise ValueError("Unsupported rebalance frequency")

        earliest_start = get_universe_start_date_cached(universe, weighting, factor_set=factor_set)
        earliest_start_year = earliest_start.year
        default_start_year = earliest_start_year

        if not start_year_display:
            start_year_display = str(default_start_year)

        if not style_window_years_value and not style_window_value:
            style_window_years_value = "1.0"

        try:
            start_year = int(start_year_display)
        except ValueError as exc:
            raise ValueError("Invalid start year") from exc

        if start_year < earliest_start_year or start_year > current_year:
            raise ValueError("Start year out of range")

        start_date = date(start_year, 1, 1)

        df = get_universe_returns_cached(
            universe,
            weighting=weighting,
            factor_set=factor_set,
            start_date=start_date,
        )

        if df.empty:
            flash("No data returned for the requested range.", "warning")
            return render_template(
                "benchmark_style.html",
                chart_growth=chart_growth,
                chart_tracking=chart_tracking,
                weights_history=weights_history,
                weights_table=weights_table,
                metrics=metrics,
                universes=SUPPORTED_INDUSTRY_UNIVERSES,
                selected_universe=selected_universe,
                style_window_years_value=style_window_years_value,
                rebalance=rebalance,
                method=method,
                start_year_value=start_year_display,
                save_name_value=save_name_value,
                save_overwrite=save_overwrite,
                earliest_start_year=earliest_start_year or current_year,
                current_year=current_year,
            )

        style_window = _opt_int(style_window_value)
        style_window_years = _opt_float(style_window_years_value)
        if style_window is not None and style_window <= 1:
            raise ValueError("style_window must be greater than 1.")
        if style_window is None and style_window_years is not None and style_window_years <= 0:
            raise ValueError("style_window_years must be positive.")

        analysis = StyleAnalysis(df, benchmark_name="Mkt")
        run = analysis.run(
            style_window=style_window,
            style_window_years=style_window_years,
            optimize_frequency=rebalance,
            method=method,
        )

        summary = _summarize_style_run(run)
        chart_growth = summary["chart_growth"]
        chart_tracking = summary["chart_tracking"]
        weights_history = summary["weights_history"]
        weights_table = summary["weights_table"]
        metrics = summary["metrics"]
        for warning in summary["warnings"]:
            flash(warning, "warning")

        if save_name_value:
            try:
                snapshot = StyleAnalysisSnapshot(
                    name=save_name_value,
                    created_at=datetime.now(),
                    universe=universe,
                    weighting=weighting,
                    factor_set=factor_set,
                    start_date=start_date,
                    end_date=None,
                    run=run,
                    universe_data=df,
                )
                save_style_snapshot(snapshot, _style_results_dir(), overwrite=save_overwrite)
                flash(f"Saved style run '{snapshot.name}'.", "success")
                save_name_value = ""
                save_overwrite = False
            except FileExistsError:
                flash(
                    "A saved run with that name already exists. Enable overwrite to replace it.",
                    "warning",
                )
            except ValueError:
                flash("Please provide a valid name to save the run.", "warning")
            except Exception:
                flash("Unable to save the benchmark style run right now.", "danger")
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to run benchmark style analysis right now.", "danger")

    return render_template(
        "benchmark_style.html",
        chart_growth=chart_growth,
        chart_tracking=chart_tracking,
        weights_history=weights_history,
        weights_table=weights_table,
        metrics=metrics,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        style_window_years_value=style_window_years_value,
        rebalance=rebalance,
        method=method,
        start_year_value=start_year_display,
        save_name_value=save_name_value,
        save_overwrite=save_overwrite,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )


@style_bp.route("/benchmark-style/saved", methods=["GET"])
def saved_benchmark_style():
    chart_growth: Optional[dict[str, object]] = None
    chart_tracking: Optional[dict[str, object]] = None
    weights_history: Optional[dict[str, object]] = None
    weights_table: list[dict[str, object]] = []
    metrics: Optional[dict[str, object]] = None
    snapshot_meta: Optional[dict[str, object]] = None

    selected_key = (request.args.get("run") or "").strip()
    saved_runs = []

    try:
        results_dir = _style_results_dir()
        saved_runs = list_style_snapshots(results_dir)

        if selected_key:
            snapshot = load_style_snapshot(snapshot_path(results_dir, selected_key))
            summary = _summarize_style_run(snapshot.run)
            chart_growth = summary["chart_growth"]
            chart_tracking = summary["chart_tracking"]
            weights_history = summary["weights_history"]
            weights_table = summary["weights_table"]
            metrics = summary["metrics"]
            for warning in summary["warnings"]:
                flash(warning, "warning")

            snapshot_meta = {
                "name": snapshot.name,
                "created_at": snapshot.created_at.strftime("%Y-%m-%d %H:%M"),
                "universe": snapshot.universe,
                "weighting": snapshot.weighting,
                "factor_set": snapshot.factor_set,
                "start_date": (
                    snapshot.start_date.strftime("%Y-%m-%d") if snapshot.start_date else None
                ),
                "end_date": snapshot.end_date.strftime("%Y-%m-%d") if snapshot.end_date else None,
            }
    except FileNotFoundError:
        flash("Saved style run not found.", "warning")
    except Exception:
        flash("Unable to load saved style runs right now.", "danger")

    return render_template(
        "benchmark_style_saved.html",
        saved_runs=saved_runs,
        selected_key=selected_key,
        snapshot_meta=snapshot_meta,
        chart_growth=chart_growth,
        chart_tracking=chart_tracking,
        weights_history=weights_history,
        weights_table=weights_table,
        metrics=metrics,
    )


@style_bp.route("/benchmark-style/saved/<run_key>/view", methods=["GET"])
def view_saved_benchmark_style(run_key: str):
    try:
        results_dir = _style_results_dir()
        snapshot = load_style_snapshot(snapshot_path(results_dir, run_key))
    except FileNotFoundError:
        flash("Saved style run not found.", "warning")
        return redirect(url_for("style.saved_benchmark_style"))
    except Exception:
        flash("Unable to load the saved run right now.", "danger")
        return redirect(url_for("style.saved_benchmark_style"))

    params: dict[str, object] = {
        "universe": snapshot.universe,
        "start_year": snapshot.start_date.year if snapshot.start_date else None,
        "style_window_years": snapshot.run.params.get("style_window_years"),
        "rebalance": snapshot.run.params.get("optimize_frequency"),
        "method": snapshot.run.params.get("method"),
    }

    window_years = snapshot.run.params.get("style_window_years")
    if window_years is None:
        params["style_window"] = snapshot.run.params.get("style_window")
    else:
        params["style_window_years"] = f"{float(window_years):.4f}".rstrip("0").rstrip(".")

    params = {key: value for key, value in params.items() if value is not None}
    return redirect(url_for("style.benchmark_style", **params))


@style_bp.route("/benchmark-style/saved/<run_key>/delete", methods=["POST"])
def delete_saved_benchmark_style(run_key: str):
    if not request.form.get("confirm"):
        flash("Please confirm deletion before removing a saved run.", "warning")
        return redirect(url_for("style.saved_benchmark_style", run=run_key))

    try:
        results_dir = _style_results_dir()
        path = snapshot_path(results_dir, run_key)
        if not path.exists():
            flash("Saved style run not found.", "warning")
        else:
            try:
                snapshot = load_style_snapshot(path)
                display_name = snapshot.name
            except Exception:
                display_name = run_key
            path.unlink()
            flash(f"Deleted saved run '{display_name}'.", "success")
    except Exception:
        flash("Unable to delete the saved run right now.", "danger")

    return redirect(url_for("style.saved_benchmark_style"))
