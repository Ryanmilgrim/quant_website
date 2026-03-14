from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from toolkit.analysis.factor_analysis import (
    FactorModel,
    FactorRun,
    annualize_vol,
    covariance_to_correlation_df,
)
from toolkit.analysis.factor_storage import (
    FactorAnalysisSnapshot,
    list_factor_snapshots,
    load_factor_snapshot,
    save_factor_snapshot,
)
from toolkit.analysis.style_storage import snapshot_path
from website.lib.data import SUPPORTED_INDUSTRY_UNIVERSES
from website.web.services.universe_cache import (
    get_universe_returns_cached,
    get_universe_start_date_cached,
)

factor_bp = Blueprint("factor", __name__, url_prefix="/factor")


def _factor_results_dir() -> Path:
    root = Path(current_app.instance_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_results" / "factor_analysis"
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


# ---------------------------------------------------------------------------
# Plotly payload builders
# ---------------------------------------------------------------------------

def _dates_to_strings(idx: pd.Index) -> list[str]:
    return [dt.strftime("%Y-%m-%d") for dt in idx]


def _safe_list(arr) -> list:
    """Convert array to list, replacing NaN/Inf with None for JSON safety."""
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    return [None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in arr]


def _safe_nested_list(arr_2d) -> list[list]:
    """Convert 2-D array to nested list, replacing NaN/Inf with None."""
    if hasattr(arr_2d, "tolist"):
        arr_2d = arr_2d.tolist()
    return [_safe_list(row) for row in arr_2d]


def _safe_float(v) -> object:
    """Return float or None if NaN/Inf."""
    f = float(v)
    return None if (np.isnan(f) or np.isinf(f)) else f


def _safe_meta_value(v):
    """Serialize a meta value to a JSON-safe type."""
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _thin(n: int, max_points: int = 2500) -> int:
    """Return step size to thin *n* points down to at most *max_points*."""
    return max(1, n // max_points)


def _summarize_factor_run(run: FactorRun) -> dict:
    """Build all Plotly JSON payloads from a FactorRun."""
    meta = run.meta
    ev = run.results.get("evaluation")

    # --- Meta ---
    meta_payload = {k: _safe_meta_value(v) for k, v in meta.items()}

    # --- Evaluation summary ---
    eval_summary = {}
    train_end = None
    if isinstance(ev, dict):
        eval_summary = ev.get("summary", {})
        # Serialize nested dicts (each value is a dict with mean/std/min/max)
        eval_summary = {
            k: {sk: _safe_float(sv) for sk, sv in v.items()}
            if isinstance(v, dict)
            else _safe_float(v)
            for k, v in eval_summary.items()
        }
        train_end = ev.get("params", {}).get("train_end")

    train_end_str = (
        train_end.strftime("%Y-%m-%d") if hasattr(train_end, "strftime") else None
    )

    # --- Beta heatmap ---
    betas = run.results.get("betas_ordered", run.beta_loadings)
    betas_arr = betas.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(betas_arr), 98)) if np.isfinite(betas_arr).any() else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    beta_heatmap = {
        "z": _safe_nested_list(betas_arr),
        "x": [str(c) for c in betas.columns],
        "y": [str(r) for r in betas.index],
        "zmin": round(-vmax, 4),
        "zmax": round(vmax, 4),
    }

    # --- Factor correlation heatmap ---
    factor_cov = run.results.get("factor_cov_forecast")
    factor_corr_heatmap = None
    if factor_cov is not None:
        corr = covariance_to_correlation_df(factor_cov, clip=True)
        factor_corr_heatmap = {
            "z": _safe_nested_list(corr.to_numpy(dtype=float).round(4)),
            "x": [str(c) for c in corr.columns],
            "y": [str(r) for r in corr.index],
            "zmin": -1,
            "zmax": 1,
        }

    # --- Evaluation timeseries-based charts ---
    agg_vol_backtest = None
    vol_regression_scatter = None
    per_asset_vol = None
    per_asset_conf = None
    per_asset_corr = None

    if isinstance(ev, dict):
        ts = ev.get("timeseries", {})

        # Aggregate vol backtest
        pred_avg = ts.get("pred_avg_vol_ann", pd.Series(dtype=float))
        real_avg = ts.get("real_avg_vol_ann", pd.Series(dtype=float))
        if not pred_avg.empty and not real_avg.empty:
            idx = pred_avg.index.intersection(real_avg.index)
            step = _thin(len(idx))
            idx = idx[::step]
            dates = _dates_to_strings(idx)
            agg_vol_backtest = {
                "series": [
                    {"name": "Predicted (avg)", "x": dates, "y": _safe_list(pred_avg.loc[idx].round(6))},
                    {"name": "Trailing (avg)", "x": dates, "y": _safe_list(real_avg.loc[idx].round(6))},
                ],
                "y_axis_title": "Annualized volatility",
                "train_end": train_end_str,
            }

        # Vol regression scatter
        pred_vol = ts.get("pred_vol", pd.DataFrame())
        real_vol = ts.get("real_vol", pd.DataFrame())
        if not pred_vol.empty and not real_vol.empty:
            idx = pred_vol.index.intersection(real_vol.index)
            if train_end is not None:
                idx = idx[idx >= pd.Timestamp(train_end)]
            pv_ann = annualize_vol(pred_vol.loc[idx])
            rv_ann = annualize_vol(real_vol.loc[idx])
            x_vals = rv_ann.mean(axis=0)
            y_vals = pv_ann.mean(axis=0)
            scatter_df = pd.concat(
                [x_vals.rename("real"), y_vals.rename("pred")], axis=1
            ).dropna()

            if not scatter_df.empty:
                xv = scatter_df["real"].to_numpy(dtype=float)
                yv = scatter_df["pred"].to_numpy(dtype=float)
                b, a = np.polyfit(xv, yv, deg=1)
                y_hat = a + b * xv
                ss_res = float(np.sum((yv - y_hat) ** 2))
                ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                lo = float(np.nanmin(np.r_[xv, yv]))
                hi = float(np.nanmax(np.r_[xv, yv]))

                vol_regression_scatter = {
                    "x": _safe_list(xv.round(6)),
                    "y": _safe_list(yv.round(6)),
                    "labels": [str(n) for n in scatter_df.index],
                    "regression": {
                        "a": _safe_float(a),
                        "b": _safe_float(b),
                        "r2": _safe_float(r2),
                    },
                    "identity_line": {"lo": _safe_float(lo), "hi": _safe_float(hi)},
                }

            # Per-asset vol backtest
            assets = [str(c) for c in pred_vol.columns]
            pred_vol_ann = annualize_vol(pred_vol)
            real_vol_ann = annualize_vol(real_vol)
            vol_idx = pred_vol_ann.index.intersection(real_vol_ann.index)
            step = _thin(len(vol_idx))
            vol_idx = vol_idx[::step]
            vol_dates = _dates_to_strings(vol_idx)
            pa_vol_data = {}
            for asset in assets:
                pa_vol_data[asset] = {
                    "x": vol_dates,
                    "pred": _safe_list(pred_vol_ann.loc[vol_idx, asset].round(6)),
                    "real": _safe_list(real_vol_ann.loc[vol_idx, asset].round(6)),
                }
            per_asset_vol = {
                "assets": assets,
                "train_end": train_end_str,
                "data": pa_vol_data,
            }

        # Per-asset confidence bands
        resid = run.results.get("resid", pd.DataFrame())
        resid_cond_var = run.results.get("resid_cond_var", pd.DataFrame())
        if not resid.empty and not resid_cond_var.empty:
            assets = [str(c) for c in resid.columns]
            pa_conf_data = {}
            for asset in assets:
                r = resid[asset].astype(float)
                s = np.sqrt(resid_cond_var[asset].astype(float).clip(lower=0.0))
                cidx = r.index.intersection(s.index)
                step = _thin(len(cidx))
                cidx = cidx[::step]
                r = r.loc[cidx]
                s = s.loc[cidx]
                cdates = _dates_to_strings(cidx)
                pa_conf_data[asset] = {
                    "x": cdates,
                    "returns": _safe_list(r.round(6)),
                    "upper_2s": _safe_list((2.0 * s).round(6)),
                    "lower_2s": _safe_list((-2.0 * s).round(6)),
                    "upper_3s": _safe_list((3.0 * s).round(6)),
                    "lower_3s": _safe_list((-3.0 * s).round(6)),
                }
            per_asset_conf = {
                "assets": assets,
                "data": pa_conf_data,
            }

        # Per-asset correlation backtest
        pred_corr = ts.get("pred_corr_to_agg", pd.DataFrame())
        real_corr = ts.get("real_corr_to_agg", pd.DataFrame())
        if not pred_corr.empty and not real_corr.empty:
            assets = [str(c) for c in pred_corr.columns]
            corr_idx = pred_corr.index.intersection(real_corr.index)
            step = _thin(len(corr_idx))
            corr_idx = corr_idx[::step]
            corr_dates = _dates_to_strings(corr_idx)
            pa_corr_data = {}
            for asset in assets:
                pa_corr_data[asset] = {
                    "x": corr_dates,
                    "pred": _safe_list(pred_corr.loc[corr_idx, asset].round(6)),
                    "real": _safe_list(real_corr.loc[corr_idx, asset].round(6)),
                }
            per_asset_corr = {
                "assets": assets,
                "train_end": train_end_str,
                "data": pa_corr_data,
            }

    return {
        "meta": meta_payload,
        "eval_summary": eval_summary,
        "train_end": train_end_str,
        "beta_heatmap": beta_heatmap,
        "factor_corr_heatmap": factor_corr_heatmap,
        "agg_vol_backtest": agg_vol_backtest,
        "vol_regression_scatter": vol_regression_scatter,
        "per_asset_vol": per_asset_vol,
        "per_asset_conf": per_asset_conf,
        "per_asset_corr": per_asset_corr,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@factor_bp.route("/factor-analysis", methods=["GET", "POST"])
def factor_analysis():
    results: Optional[dict] = None

    params = request.values
    selected_universe = (params.get("universe") or "10").strip()
    start_year_value = (params.get("start_year") or "").strip()
    garch_dist = (params.get("garch_dist") or "t").strip()
    pca_demean = bool(params.get("pca_demean"))
    train_fraction_value = (params.get("train_fraction") or "0.7").strip()
    realized_window_value = (params.get("realized_window") or "60").strip()
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

        if garch_dist not in {"normal", "t", "skewt"}:
            raise ValueError("Unsupported GARCH distribution")

        train_fraction = float(train_fraction_value)
        if not (0.1 <= train_fraction <= 0.9):
            raise ValueError("train_fraction must be between 0.1 and 0.9")

        realized_window = int(realized_window_value)
        if realized_window < 5:
            raise ValueError("realized_window must be at least 5")

        earliest_start = get_universe_start_date_cached(universe, weighting, factor_set=factor_set)
        earliest_start_year = earliest_start.year
        default_start_year = earliest_start_year

        if not start_year_display:
            start_year_display = str(default_start_year)

        try:
            start_year = int(start_year_display)
        except ValueError as exc:
            raise ValueError("Invalid start year") from exc

        if start_year < earliest_start_year or start_year > current_year:
            raise ValueError("Start year out of range")

        start_date = date(start_year, 1, 1)

        if request.method == "POST":
            df = get_universe_returns_cached(
                universe,
                weighting=weighting,
                factor_set=factor_set,
                start_date=start_date,
            )

            if df.empty:
                flash("No data returned for the requested range.", "warning")
            else:
                fm = FactorModel(
                    rf_name="Rf",
                    garch_dist=garch_dist,
                    pca_demean=pca_demean,
                )
                run = fm.evaluate_train_test(
                    uni=df,
                    train_fraction=train_fraction,
                    realized_window=realized_window,
                    progress=False,
                )
                results = _summarize_factor_run(run)

                if save_name_value:
                    try:
                        snapshot = FactorAnalysisSnapshot(
                            name=save_name_value,
                            created_at=datetime.now(),
                            universe=universe,
                            weighting=weighting,
                            factor_set=factor_set,
                            start_date=start_date,
                            end_date=None,
                            garch_dist=garch_dist,
                            pca_demean=pca_demean,
                            train_fraction=train_fraction,
                            realized_window=realized_window,
                            run=run,
                            universe_data=df,
                        )
                        save_factor_snapshot(
                            snapshot, _factor_results_dir(), overwrite=save_overwrite,
                        )
                        flash(f"Saved factor run '{snapshot.name}'.", "success")
                        save_name_value = ""
                        save_overwrite = False
                    except FileExistsError:
                        flash(
                            "A saved run with that name already exists. Enable overwrite to replace it.",
                            "warning",
                        )
                    except Exception:
                        flash("Unable to save the factor run right now.", "danger")
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to run factor analysis right now.", "danger")

    return render_template(
        "factor_analysis.html",
        results=results,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        start_year_value=start_year_display,
        garch_dist=garch_dist,
        pca_demean=pca_demean,
        train_fraction_value=train_fraction_value,
        realized_window_value=realized_window_value,
        save_name_value=save_name_value,
        save_overwrite=save_overwrite,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )


@factor_bp.route("/factor-analysis/saved", methods=["GET"])
def saved_factor_analysis():
    results: Optional[dict] = None
    snapshot_meta: Optional[dict] = None
    selected_key = (request.args.get("run") or "").strip()
    saved_runs = []

    try:
        results_dir = _factor_results_dir()
        saved_runs = list_factor_snapshots(results_dir)

        if selected_key:
            snapshot = load_factor_snapshot(snapshot_path(results_dir, selected_key))
            results = _summarize_factor_run(snapshot.run)
            snapshot_meta = {
                "name": snapshot.name,
                "created_at": snapshot.created_at.strftime("%Y-%m-%d %H:%M"),
                "universe": snapshot.universe,
                "weighting": snapshot.weighting,
                "factor_set": snapshot.factor_set,
                "start_date": (
                    snapshot.start_date.strftime("%Y-%m-%d") if snapshot.start_date else None
                ),
                "end_date": (
                    snapshot.end_date.strftime("%Y-%m-%d") if snapshot.end_date else None
                ),
                "garch_dist": snapshot.garch_dist,
                "pca_demean": snapshot.pca_demean,
                "train_fraction": snapshot.train_fraction,
                "realized_window": snapshot.realized_window,
            }
    except FileNotFoundError:
        flash("Saved factor run not found.", "warning")
    except Exception:
        flash("Unable to load saved factor runs right now.", "danger")

    return render_template(
        "factor_analysis_saved.html",
        saved_runs=saved_runs,
        selected_key=selected_key,
        snapshot_meta=snapshot_meta,
        results=results,
    )


@factor_bp.route("/factor-analysis/saved/<run_key>/view", methods=["GET"])
def view_saved_factor_analysis(run_key: str):
    try:
        results_dir = _factor_results_dir()
        snapshot = load_factor_snapshot(snapshot_path(results_dir, run_key))
    except FileNotFoundError:
        flash("Saved factor run not found.", "warning")
        return redirect(url_for("factor.saved_factor_analysis"))
    except Exception:
        flash("Unable to load the saved run right now.", "danger")
        return redirect(url_for("factor.saved_factor_analysis"))

    params = {
        "universe": snapshot.universe,
        "start_year": snapshot.start_date.year if snapshot.start_date else None,
        "garch_dist": snapshot.garch_dist,
        "pca_demean": "1" if snapshot.pca_demean else None,
        "train_fraction": snapshot.train_fraction,
        "realized_window": snapshot.realized_window,
    }
    params = {k: v for k, v in params.items() if v is not None}
    return redirect(url_for("factor.factor_analysis", **params))


@factor_bp.route("/factor-analysis/saved/<run_key>/delete", methods=["POST"])
def delete_saved_factor_analysis(run_key: str):
    if not request.form.get("confirm"):
        flash("Please confirm deletion before removing a saved run.", "warning")
        return redirect(url_for("factor.saved_factor_analysis", run=run_key))

    try:
        results_dir = _factor_results_dir()
        path = snapshot_path(results_dir, run_key)
        if not path.exists():
            flash("Saved factor run not found.", "warning")
        else:
            try:
                snap = load_factor_snapshot(path)
                display_name = snap.name
            except Exception:
                display_name = run_key
            path.unlink()
            flash(f"Deleted saved run '{display_name}'.", "success")
    except Exception:
        flash("Unable to delete the saved run right now.", "danger")

    return redirect(url_for("factor.saved_factor_analysis"))
