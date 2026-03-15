from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uuid

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

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
from website.lib.data import SUPPORTED_FACTOR_SETS, SUPPORTED_INDUSTRY_UNIVERSES
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


# ---------------------------------------------------------------------------
# Run cache — holds FactorRun objects so AJAX endpoints can build chart data
# on demand without re-serializing everything into the initial HTML payload.
# ---------------------------------------------------------------------------
_run_cache: dict[str, FactorRun] = {}
_MAX_CACHED_RUNS = 4


def _cache_run(run: FactorRun) -> str:
    """Store *run* and return a cache key.  Evicts oldest if full."""
    run_id = uuid.uuid4().hex[:12]
    if len(_run_cache) >= _MAX_CACHED_RUNS:
        _run_cache.pop(next(iter(_run_cache)), None)
    _run_cache[run_id] = run
    return run_id


def _summarize_factor_run(run: FactorRun) -> dict:
    """Build *lightweight* metadata payload for initial page load.

    Heavy chart data (factor vol TS, backtesting series, pairwise data) is
    NOT included — the frontend fetches it lazily via the ``/chart-data``
    AJAX endpoint.
    """
    meta = run.meta
    ev = run.results.get("evaluation")

    # --- Meta ---
    meta_payload = {k: _safe_meta_value(v) for k, v in meta.items()}

    # --- Per-asset metrics table ---
    asset_metrics = []
    r2_series = run.results.get("r2", pd.Series(dtype=float))
    asset_cond_vol_df = run.results.get("asset_cond_vol")
    resid_cond_var_df = run.results.get("resid_cond_var")
    assets_excess_full = run.results.get("assets_excess")
    resid_full = run.results.get("resid", pd.DataFrame())
    alpha_intercept = run.results.get("alpha_intercept", pd.Series(dtype=float))
    if r2_series is not None and not r2_series.empty:
        for asset in r2_series.index:
            row = {"asset": str(asset), "r2": _safe_float(r2_series.loc[asset])}
            total_var_daily = None
            if asset_cond_vol_df is not None and asset in asset_cond_vol_df.columns:
                last_vol = float(asset_cond_vol_df[asset].iloc[-1])
                total_var_daily = last_vol ** 2
                row["total_vol_ann"] = _safe_float(last_vol * float(np.sqrt(252)))
            else:
                row["total_vol_ann"] = None
            resid_var_daily = None
            if resid_cond_var_df is not None and asset in resid_cond_var_df.columns:
                resid_var_daily = float(resid_cond_var_df[asset].iloc[-1])
                row["resid_vol_ann"] = _safe_float(
                    float(np.sqrt(max(resid_var_daily, 0.0))) * float(np.sqrt(252))
                )
            else:
                row["resid_vol_ann"] = None
            if total_var_daily is not None and resid_var_daily is not None:
                sys_var = max(total_var_daily - resid_var_daily, 0.0)
                row["systematic_vol_ann"] = _safe_float(float(np.sqrt(sys_var * 252)))
                row["factor_risk_pct"] = _safe_float(
                    sys_var / total_var_daily if total_var_daily > 0 else 0.0
                )
            else:
                row["systematic_vol_ann"] = None
                row["factor_risk_pct"] = None
            if assets_excess_full is not None and asset in assets_excess_full.columns:
                row["total_return_ann"] = _safe_float(
                    float(assets_excess_full[asset].mean()) * 252
                )
            else:
                row["total_return_ann"] = None
            if (
                assets_excess_full is not None
                and asset in assets_excess_full.columns
                and not resid_full.empty
                and asset in resid_full.columns
            ):
                sys_ret = assets_excess_full[asset] - resid_full[asset]
                row["systematic_return_ann"] = _safe_float(float(sys_ret.mean()) * 252)
                row["residual_return_ann"] = _safe_float(
                    float(resid_full[asset].mean()) * 252
                )
            else:
                row["systematic_return_ann"] = None
                row["residual_return_ann"] = None
            if (
                alpha_intercept is not None
                and not alpha_intercept.empty
                and asset in alpha_intercept.index
            ):
                row["alpha_ann"] = _safe_float(
                    float(alpha_intercept.loc[asset]) * 252
                )
            else:
                row["alpha_ann"] = None
            if row.get("residual_return_ann") is not None and row.get("resid_vol_ann"):
                row["residual_sharpe"] = _safe_float(
                    row["residual_return_ann"] / row["resid_vol_ann"]
                )
            else:
                row["residual_sharpe"] = None
            asset_metrics.append(row)

    # --- Train end ---
    train_end = None
    if isinstance(ev, dict):
        train_end = ev.get("params", {}).get("train_end")
    train_end_str = (
        train_end.strftime("%Y-%m-%d") if hasattr(train_end, "strftime") else None
    )

    # --- Capability flags (tell the template which chart sections to render) ---
    has_factor_risk = (
        run.results.get("pc_cond_var") is not None
        or run.results.get("beta_loadings") is not None
    )
    has_backtest = isinstance(ev, dict) and bool(ev.get("timeseries"))

    return {
        "meta": meta_payload,
        "asset_metrics": asset_metrics,
        "train_end": train_end_str,
        "has_factor_risk": has_factor_risk,
        "has_backtest": has_backtest,
    }


# ---------------------------------------------------------------------------
# Per-section chart data builders (called lazily via AJAX)
# ---------------------------------------------------------------------------

def _build_factor_risk_data(run: FactorRun) -> dict:
    """Build factor risk chart payloads (beta heatmap, factor vol TS, etc.)."""
    ev = run.results.get("evaluation")
    train_end = None
    if isinstance(ev, dict):
        train_end = ev.get("params", {}).get("train_end")
    train_end_str = (
        train_end.strftime("%Y-%m-%d") if hasattr(train_end, "strftime") else None
    )

    # Beta heatmap
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

    # Factor correlation / covariance heatmaps
    factor_cov = run.results.get("factor_cov_forecast")
    factor_corr_heatmap = None
    factor_cov_heatmap = None
    if factor_cov is not None:
        corr = covariance_to_correlation_df(factor_cov, clip=True)
        factor_corr_heatmap = {
            "z": _safe_nested_list(corr.to_numpy(dtype=float).round(4)),
            "x": [str(c) for c in corr.columns],
            "y": [str(r) for r in corr.index],
            "zmin": -1, "zmax": 1,
        }
        cov_ann = factor_cov.to_numpy(dtype=float) * 252
        factor_cov_heatmap = {
            "z": _safe_nested_list(np.round(cov_ann, 6)),
            "x": [str(c) for c in factor_cov.columns],
            "y": [str(r) for r in factor_cov.index],
        }

    # Factor volatility timeseries
    factor_vol_ts = None
    pc_cond_var = run.results.get("pc_cond_var")
    eigen_vectors = run.results.get("eigen_vectors")
    if pc_cond_var is not None and eigen_vectors is not None:
        ev_mat = eigen_vectors.to_numpy(dtype=float)
        fv_idx = pc_cond_var.index
        fv_dates = _dates_to_strings(fv_idx)
        factor_names = [str(f) for f in eigen_vectors.index]
        n_f = len(factor_names)

        V_sq = ev_mat ** 2
        h_pc_arr = pc_cond_var.to_numpy(dtype=float)
        factor_var = h_pc_arr @ V_sq.T
        factor_vol = np.sqrt(np.clip(factor_var, 0, None)) * float(np.sqrt(252))

        fv_vol_data = {}
        for i, fname in enumerate(factor_names):
            fv_vol_data[fname] = _safe_list(np.round(factor_vol[:, i], 6))

        h_clipped = np.clip(h_pc_arr, 0.0, None)
        Vh = ev_mat[np.newaxis, :, :] * h_clipped[:, np.newaxis, :]
        cov_all = np.einsum("tkp,jp->tkj", Vh, ev_mat)
        cov_all = 0.5 * (cov_all + np.swapaxes(cov_all, 1, 2))
        diag_vol = np.sqrt(np.clip(np.diagonal(cov_all, axis1=1, axis2=2), 1e-30, None))
        denom = diag_vol[:, :, np.newaxis] * diag_vol[:, np.newaxis, :]
        corr_all = np.clip(cov_all / denom, -1.0, 1.0)

        fv_corr_data: dict[str, dict[str, list]] = {}
        for fi in range(n_f):
            fname = factor_names[fi]
            fv_corr_data[fname] = {}
            for fj in range(n_f):
                if fi == fj:
                    continue
                fv_corr_data[fname][factor_names[fj]] = _safe_list(
                    np.round(corr_all[:, fi, fj], 6)
                )

        cov_ann = cov_all * 252
        fv_cov_data: dict[str, list] = {}
        for fi in range(n_f):
            fv_cov_data[f"Var({factor_names[fi]})"] = _safe_list(
                np.round(cov_ann[:, fi, fi], 8)
            )
            for fj in range(fi + 1, n_f):
                fv_cov_data[f"Cov({factor_names[fi]}, {factor_names[fj]})"] = _safe_list(
                    np.round(cov_ann[:, fi, fj], 8)
                )

        factor_vol_ts = {
            "factors": factor_names,
            "x": fv_dates,
            "vol_data": fv_vol_data,
            "corr_data": fv_corr_data,
            "cov_data": fv_cov_data,
            "train_end": train_end_str,
        }

    return {
        "beta_heatmap": beta_heatmap,
        "factor_corr_heatmap": factor_corr_heatmap,
        "factor_cov_heatmap": factor_cov_heatmap,
        "factor_vol_ts": factor_vol_ts,
    }


def _build_backtest_data(run: FactorRun, *, sub: str | None = None) -> dict:
    """Build backtesting chart payloads.

    When *sub* is given, only the requested metric is built:
      ``"volatility"`` — agg_vol_backtest + per_asset_vol
      ``"correlation"`` — per_asset_corr + pairwise_corr
      ``"covariance"``  — per_asset_cov + pairwise_cov
      ``"returns"``     — per_asset_conf (std residuals + total returns)
    When *sub* is ``None``, everything is built (backward-compat).
    """
    ev = run.results.get("evaluation")
    if not isinstance(ev, dict):
        return {}

    train_end = ev.get("params", {}).get("train_end")
    train_end_str = (
        train_end.strftime("%Y-%m-%d") if hasattr(train_end, "strftime") else None
    )
    ts = ev.get("timeseries", {})

    result: dict = {}
    build_all = sub is None

    # --- VOLATILITY ---
    if build_all or sub == "volatility":
        agg_vol_backtest = None
        per_asset_vol = None

        pred_avg = ts.get("pred_avg_vol_ann", pd.Series(dtype=float))
        real_avg = ts.get("real_avg_vol_ann", pd.Series(dtype=float))
        if not pred_avg.empty and not real_avg.empty:
            idx = pred_avg.index.intersection(real_avg.index)
            dates = _dates_to_strings(idx)
            agg_vol_backtest = {
                "series": [
                    {"name": "Predicted (avg)", "x": dates, "y": _safe_list(pred_avg.loc[idx].round(6))},
                    {"name": "Trailing (avg)", "x": dates, "y": _safe_list(real_avg.loc[idx].round(6))},
                ],
                "y_axis_title": "Annualized volatility",
                "train_end": train_end_str,
            }

        pred_vol = ts.get("pred_vol", pd.DataFrame())
        real_vol = ts.get("real_vol", pd.DataFrame())
        if not pred_vol.empty and not real_vol.empty:
            assets = [str(c) for c in pred_vol.columns]
            pred_vol_ann = annualize_vol(pred_vol)
            real_vol_ann = annualize_vol(real_vol)
            vol_idx = pred_vol_ann.index.intersection(real_vol_ann.index)
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

        result["agg_vol_backtest"] = agg_vol_backtest
        result["per_asset_vol"] = per_asset_vol

    # --- RETURNS / RESIDUALS ---
    if build_all or sub == "returns":
        per_asset_conf = None
        resid = run.results.get("resid", pd.DataFrame())
        resid_cond_var_ev = run.results.get("resid_cond_var", pd.DataFrame())
        assets_excess_df = run.results.get("assets_excess")
        asset_cond_vol_df = run.results.get("asset_cond_vol")
        if not resid.empty and not resid_cond_var_ev.empty:
            assets = [str(c) for c in resid.columns]
            pa_conf_data = {}
            for asset in assets:
                r = resid[asset].astype(float)
                s = np.sqrt(resid_cond_var_ev[asset].astype(float).clip(lower=0.0))
                cidx = r.index.intersection(s.index)
                r = r.loc[cidx]
                s = s.loc[cidx]
                std_resid = (r / s.replace(0.0, np.nan)).fillna(0.0)
                cdates = _dates_to_strings(cidx)
                n_obs = len(std_resid)
                pct_in_2s = float((std_resid.abs() <= 2.0).sum() / n_obs) if n_obs else 0.0
                pct_in_3s = float((std_resid.abs() <= 3.0).sum() / n_obs) if n_obs else 0.0
                entry: dict = {
                    "x": cdates,
                    "std_resid": _safe_list(std_resid.round(4)),
                    "pct_in_2s": round(pct_in_2s * 100, 1),
                    "pct_in_3s": round(pct_in_3s * 100, 1),
                }
                if (
                    assets_excess_df is not None
                    and asset in assets_excess_df.columns
                ):
                    ar = assets_excess_df[asset].astype(float).reindex(cidx)
                    entry["asset_returns"] = _safe_list(ar.round(6))
                if (
                    asset_cond_vol_df is not None
                    and asset in asset_cond_vol_df.columns
                ):
                    cond_vol = asset_cond_vol_df[asset].astype(float).reindex(cidx)
                    entry["upper_2s"] = _safe_list((2.0 * cond_vol).round(6))
                    entry["lower_2s"] = _safe_list((-2.0 * cond_vol).round(6))
                pa_conf_data[asset] = entry
            per_asset_conf = {
                "assets": assets,
                "data": pa_conf_data,
            }
        result["per_asset_conf"] = per_asset_conf

    # --- CORRELATION ---
    if build_all or sub == "correlation":
        per_asset_corr = None
        pairwise_corr = None

        pred_corr_agg = ts.get("pred_corr_to_agg", pd.DataFrame())
        real_corr_agg = ts.get("real_corr_to_agg", pd.DataFrame())
        if not pred_corr_agg.empty and not real_corr_agg.empty:
            assets = [str(c) for c in pred_corr_agg.columns]
            corr_idx = pred_corr_agg.index.intersection(real_corr_agg.index)
            corr_dates = _dates_to_strings(corr_idx)
            pa_corr_data = {}
            for asset in assets:
                pa_corr_data[asset] = {
                    "x": corr_dates,
                    "pred": _safe_list(pred_corr_agg.loc[corr_idx, asset].round(6)),
                    "real": _safe_list(real_corr_agg.loc[corr_idx, asset].round(6)),
                }
            per_asset_corr = {
                "assets": assets,
                "train_end": train_end_str,
                "data": pa_corr_data,
            }

        pred_pw = ts.get("pred_corr_pairwise", {})
        real_pw = ts.get("real_corr_pairwise", {})
        if pred_pw and real_pw:
            pw_pairs = {}
            sample_key = next(iter(pred_pw))
            pw_idx = pred_pw[sample_key].index
            pw_dates = _dates_to_strings(pw_idx)
            for key in pred_pw:
                if key not in real_pw:
                    continue
                pw_pairs[key] = {
                    "x": pw_dates,
                    "pred": _safe_list(pred_pw[key].loc[pw_idx].round(6)),
                    "real": _safe_list(real_pw[key].loc[pw_idx].round(6)),
                }
            pairwise_corr = {
                "pairs": pw_pairs,
                "train_end": train_end_str,
            }

        result["per_asset_corr"] = per_asset_corr
        result["pairwise_corr"] = pairwise_corr

    # --- COVARIANCE ---
    if build_all or sub == "covariance":
        per_asset_cov = None
        pairwise_cov = None

        pred_cov_agg_df = ts.get("pred_cov_to_agg", pd.DataFrame())
        real_cov_agg_df = ts.get("real_cov_to_agg", pd.DataFrame())
        if not pred_cov_agg_df.empty and not real_cov_agg_df.empty:
            assets = [str(c) for c in pred_cov_agg_df.columns]
            cov_idx = pred_cov_agg_df.index.intersection(real_cov_agg_df.index)
            cov_dates = _dates_to_strings(cov_idx)
            pa_cov_data = {}
            for asset in assets:
                pa_cov_data[asset] = {
                    "x": cov_dates,
                    "pred": _safe_list(pred_cov_agg_df.loc[cov_idx, asset].round(8)),
                    "real": _safe_list(real_cov_agg_df.loc[cov_idx, asset].round(8)),
                }
            per_asset_cov = {
                "assets": assets,
                "train_end": train_end_str,
                "data": pa_cov_data,
            }

        pred_cov_pw = ts.get("pred_cov_pairwise", {})
        real_cov_pw = ts.get("real_cov_pairwise", {})
        if pred_cov_pw and real_cov_pw:
            cov_pw_pairs = {}
            sample_key = next(iter(pred_cov_pw))
            cov_pw_idx = pred_cov_pw[sample_key].index
            cov_pw_dates = _dates_to_strings(cov_pw_idx)
            for key in pred_cov_pw:
                if key not in real_cov_pw:
                    continue
                cov_pw_pairs[key] = {
                    "x": cov_pw_dates,
                    "pred": _safe_list(pred_cov_pw[key].loc[cov_pw_idx].round(8)),
                    "real": _safe_list(real_cov_pw[key].loc[cov_pw_idx].round(8)),
                }
            pairwise_cov = {
                "pairs": cov_pw_pairs,
                "train_end": train_end_str,
            }

        result["per_asset_cov"] = per_asset_cov
        result["pairwise_cov"] = pairwise_cov

    return result


# ---------------------------------------------------------------------------
# AJAX endpoint — lazily serve chart data for a cached run
# ---------------------------------------------------------------------------

@factor_bp.route("/factor-analysis/chart-data/<section>")
def factor_chart_data(section: str):
    """Return JSON chart payload for *section* (``factor_risk`` or ``backtest``).

    Query-string must include ``run_id`` matching a key in ``_run_cache``.
    """
    run_id = request.args.get("run_id", "")
    run = _run_cache.get(run_id)
    if run is None:
        return jsonify({"error": "Run not found or expired. Please re-run the analysis."}), 404

    if section == "factor_risk":
        return jsonify(_build_factor_risk_data(run))
    elif section == "backtest":
        sub = request.args.get("sub", "")
        return jsonify(_build_backtest_data(run, sub=sub or None))
    else:
        return jsonify({"error": f"Unknown section: {section}"}), 400


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@factor_bp.route("/factor-analysis", methods=["GET", "POST"])
def factor_analysis():
    results: Optional[dict] = None
    run_id: Optional[str] = None

    params = request.values
    selected_universe = (params.get("universe") or "10").strip()
    start_year_value = (params.get("start_year") or "").strip()
    garch_dist = (params.get("garch_dist") or "t").strip()
    factor_set = (params.get("factor_set") or "ff3").strip()
    train_fraction_value = (params.get("train_fraction") or "0.7").strip()
    realized_window_value = (params.get("realized_window") or "60").strip()
    save_name_value = (params.get("save_name") or "").strip()
    save_overwrite = bool(params.get("save_overwrite"))

    weighting = "value"
    current_year = date.today().year
    earliest_start_year: Optional[int] = None
    start_year_display: Optional[str] = start_year_value

    # --- Auto-load default saved run on GET ---
    if request.method == "GET" and results is None:
        try:
            results_dir = _factor_results_dir()
            default_path = snapshot_path(results_dir, "Alpha_Example")
            if default_path.exists():
                snap = load_factor_snapshot(default_path)
                results = _summarize_factor_run(snap.run)
                run_id = _cache_run(snap.run)
                selected_universe = str(snap.universe)
                garch_dist = snap.garch_dist or garch_dist
                factor_set = snap.factor_set or factor_set
                train_fraction_value = str(snap.train_fraction)
                realized_window_value = str(snap.realized_window)
                if snap.start_date:
                    start_year_value = str(snap.start_date.year)
                    start_year_display = start_year_value
        except Exception:
            pass  # silently fall back to empty form

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if garch_dist not in {"normal", "t", "skewt"}:
            raise ValueError("Unsupported GARCH distribution")

        if factor_set not in SUPPORTED_FACTOR_SETS:
            raise ValueError("Unsupported factor set")

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
                    pca_demean=False,
                )
                run = fm.evaluate_train_test(
                    uni=df,
                    train_fraction=train_fraction,
                    realized_window=realized_window,
                    progress=False,
                )
                results = _summarize_factor_run(run)
                run_id = _cache_run(run)

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
                            pca_demean=False,
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
        run_id=run_id,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        start_year_value=start_year_display,
        garch_dist=garch_dist,
        factor_set=factor_set,
        train_fraction_value=train_fraction_value,
        realized_window_value=realized_window_value,
        save_name_value=save_name_value,
        save_overwrite=save_overwrite,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )


@factor_bp.route("/factor-analysis/saved", methods=["GET"])
def saved_factor_analysis():
    selected_key = (request.args.get("run") or "").strip()
    saved_runs = []

    try:
        results_dir = _factor_results_dir()
        saved_runs = list_factor_snapshots(results_dir)

        if selected_key:
            return redirect(url_for("factor.view_saved_factor_analysis", run_key=selected_key))
    except FileNotFoundError:
        flash("Saved factor run not found.", "warning")
    except Exception:
        flash("Unable to load saved factor runs right now.", "danger")

    return render_template(
        "factor_analysis_saved.html",
        saved_runs=saved_runs,
        selected_key=selected_key,
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

    run = snapshot.run
    results = _summarize_factor_run(run)
    run_id = _cache_run(run)

    current_year = date.today().year
    earliest_start_year = current_year
    start_year_value = ""
    if snapshot.start_date:
        start_year_value = str(snapshot.start_date.year)

    try:
        earliest_start = get_universe_start_date_cached(
            snapshot.universe,
            snapshot.weighting,
            factor_set=snapshot.factor_set,
        )
        earliest_start_year = earliest_start.year
    except Exception:
        pass

    return render_template(
        "factor_analysis.html",
        results=results,
        run_id=run_id,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=str(snapshot.universe),
        start_year_value=start_year_value,
        garch_dist=snapshot.garch_dist or "t",
        factor_set=snapshot.factor_set or "ff3",
        train_fraction_value=str(snapshot.train_fraction),
        realized_window_value=str(snapshot.realized_window),
        save_name_value="",
        save_overwrite=False,
        earliest_start_year=earliest_start_year,
        current_year=current_year,
    )


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
