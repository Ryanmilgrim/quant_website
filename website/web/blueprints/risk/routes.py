from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uuid

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from toolkit.analysis.risk_model import (
    RiskModel,
    RiskModelRun,
    annualize_vol,
    covariance_to_correlation_df,
)
from toolkit.analysis.risk_storage import (
    RiskModelSnapshot,
    list_factor_snapshots as list_risk_snapshots,
    load_factor_snapshot as load_risk_snapshot,
    save_factor_snapshot as save_risk_snapshot,
)
from toolkit.analysis.style_storage import snapshot_path
from website.lib.data import SUPPORTED_FACTOR_SETS, SUPPORTED_INDUSTRY_UNIVERSES
from website.web.services.universe_cache import (
    get_universe_returns_cached,
    get_universe_start_date_cached,
)

risk_bp = Blueprint("risk", __name__, url_prefix="/risk")


def _risk_results_dir() -> Path:
    root = Path(current_app.instance_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_results" / "risk_model"
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
# Run cache — holds RiskModelRun objects so AJAX endpoints can build chart data
# on demand without re-serializing everything into the initial HTML payload.
# ---------------------------------------------------------------------------
_run_cache: dict[str, RiskModelRun] = {}
_run_meta_cache: dict[str, dict] = {}
_MAX_CACHED_RUNS = 4


def _cache_run(run: RiskModelRun, meta: Optional[dict] = None) -> str:
    """Store *run* and return a cache key.  Evicts oldest if full."""
    run_id = uuid.uuid4().hex[:12]
    if len(_run_cache) >= _MAX_CACHED_RUNS:
        evict_id = next(iter(_run_cache))
        _run_cache.pop(evict_id, None)
        _run_meta_cache.pop(evict_id, None)
    _run_cache[run_id] = run
    if meta is not None:
        _run_meta_cache[run_id] = meta
    return run_id


def _summarize_factor_run(run: RiskModelRun) -> dict:
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

    has_kalman_betas = "beta_ts" in run.results

    return {
        "meta": meta_payload,
        "asset_metrics": asset_metrics,
        "train_end": train_end_str,
        "has_factor_risk": has_factor_risk,
        "has_backtest": has_backtest,
        "has_kalman_betas": has_kalman_betas,
    }


# ---------------------------------------------------------------------------
# Per-section chart data builders (called lazily via AJAX)
# ---------------------------------------------------------------------------

def _build_factor_risk_data(run: RiskModelRun) -> dict:
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


def _build_backtest_data(run: RiskModelRun, *, sub: str | None = None) -> dict:
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
                # Raw residuals
                entry["raw_resid"] = _safe_list(r.round(6))
                if (
                    asset_cond_vol_df is not None
                    and asset in asset_cond_vol_df.columns
                ):
                    cond_vol = asset_cond_vol_df[asset].astype(float).reindex(cidx)
                    entry["upper_99"] = _safe_list((2.576 * cond_vol).round(6))
                    entry["lower_99"] = _safe_list((-2.576 * cond_vol).round(6))
                # Residual 99% CI from residual GARCH
                entry["resid_upper_99"] = _safe_list((2.576 * s).round(6))
                entry["resid_lower_99"] = _safe_list((-2.576 * s).round(6))
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


def _build_kalman_beta_data(run: RiskModelRun) -> dict:
    """Build time-varying Kalman beta chart payloads (by-asset and by-factor views)."""
    beta_ts = run.results.get("beta_ts", {})
    beta_se = run.results.get("beta_se", {})
    rolling_ols_ts = run.results.get("rolling_ols_ts", {})

    ev = run.results.get("evaluation")
    train_end = None
    if isinstance(ev, dict):
        train_end = ev.get("params", {}).get("train_end")
    train_end_str = (
        train_end.strftime("%Y-%m-%d") if hasattr(train_end, "strftime") else None
    )

    asset_names = sorted(beta_ts.keys())
    by_asset: dict = {}
    factors_by_asset: dict = {}
    all_factors_set: set[str] = set()
    by_factor_raw: dict[str, dict[str, dict]] = {}  # factor -> {asset -> {x, values}}

    for asset in asset_names:
        df = beta_ts[asset]  # DataFrame(T x k_selected)
        se_df = beta_se.get(asset)
        factors = [str(c) for c in df.columns]
        factors_by_asset[asset] = factors
        all_factors_set.update(factors)

        step = _thin(len(df))
        thinned = df.iloc[::step]
        dates = _dates_to_strings(thinned.index)

        filtered: dict[str, list] = {}
        upper_2se: dict[str, list] = {}
        lower_2se: dict[str, list] = {}

        for factor in factors:
            beta_vals = thinned[factor].to_numpy(dtype=float)
            filtered[factor] = _safe_list(np.round(beta_vals, 6))

            if se_df is not None and factor in se_df.columns:
                se_vals = se_df[factor].iloc[::step].to_numpy(dtype=float)
                upper_2se[factor] = _safe_list(np.round(beta_vals + 2.0 * se_vals, 6))
                lower_2se[factor] = _safe_list(np.round(beta_vals - 2.0 * se_vals, 6))
            else:
                upper_2se[factor] = _safe_list(np.round(beta_vals, 6))
                lower_2se[factor] = _safe_list(np.round(beta_vals, 6))

            # Accumulate for by-factor view
            if factor not in by_factor_raw:
                by_factor_raw[factor] = {}
            by_factor_raw[factor][asset] = {
                "x": dates,
                "values": filtered[factor],
            }

        rolling_ols: dict[str, list] = {}
        if rolling_ols_ts and asset in rolling_ols_ts:
            roll_df = rolling_ols_ts[asset]
            roll_thinned = roll_df.iloc[::step]
            for factor in factors:
                if factor in roll_thinned.columns:
                    rolling_ols[factor] = _safe_list(roll_thinned[factor].round(4))

        by_asset[asset] = {
            "x": dates,
            "filtered": filtered,
            "upper_2se": upper_2se,
            "lower_2se": lower_2se,
            "rolling_ols": rolling_ols,
        }

    # Build by-factor payload
    all_factors = sorted(all_factors_set)
    by_factor: dict = {}
    for factor in all_factors:
        asset_data = by_factor_raw.get(factor, {})
        if not asset_data:
            continue
        # Use dates from the first asset that has this factor
        sample_asset = next(iter(asset_data))
        x = asset_data[sample_asset]["x"]
        betas: dict[str, list] = {}
        for a in sorted(asset_data.keys()):
            betas[a] = asset_data[a]["values"]
        by_factor[factor] = {"x": x, "betas": betas}

    all_factors_list = sorted(set(str(c) for c in run.beta_loadings.columns))

    return {
        "assets": asset_names,
        "factors_by_asset": factors_by_asset,
        "all_factors": all_factors_list,
        "by_asset": by_asset,
        "factors": all_factors,
        "by_factor": by_factor,
        "train_end": train_end_str,
    }


# ---------------------------------------------------------------------------
# AJAX endpoint — lazily serve chart data for a cached run
# ---------------------------------------------------------------------------

@risk_bp.route("/risk-model/chart-data/<section>")
def risk_chart_data(section: str):
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
    elif section == "kalman_betas":
        return jsonify(_build_kalman_beta_data(run))
    else:
        return jsonify({"error": f"Unknown section: {section}"}), 400


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@risk_bp.route("/risk-model", methods=["GET", "POST"])
def risk_model():
    results: Optional[dict] = None
    run_id: Optional[str] = None

    params = request.values
    selected_universe = (params.get("universe") or "10").strip()
    start_year_value = (params.get("start_year") or "").strip()
    garch_dist = (params.get("garch_dist") or "t").strip()
    factor_set = (params.get("factor_set") or "ff3").strip()
    train_fraction_value = (params.get("train_fraction") or "0.7").strip()
    realized_window_value = (params.get("realized_window") or "60").strip()
    tstat_cutoff_value = (params.get("tstat_cutoff") or "4.0").strip()
    kalman_q_scale_value = (params.get("kalman_q_scale") or "0.01").strip()

    weighting = "value"
    current_year = date.today().year
    earliest_start_year: Optional[int] = None
    start_year_display: Optional[str] = start_year_value

    # --- Auto-load default saved run on GET ---
    if request.method == "GET" and results is None:
        try:
            results_dir = _risk_results_dir()
            default_path = snapshot_path(results_dir, "Alpha_Example")
            if default_path.exists():
                snap = load_risk_snapshot(default_path)
                results = _summarize_factor_run(snap.run)
                run_id = _cache_run(snap.run, meta={
                    "universe": snap.universe,
                    "weighting": snap.weighting,
                    "factor_set": snap.factor_set,
                    "start_date": snap.start_date,
                    "garch_dist": snap.garch_dist,
                    "pca_demean": snap.pca_demean,
                    "train_fraction": snap.train_fraction,
                    "realized_window": snap.realized_window,
                    "universe_data": snap.universe_data,
                })
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

        tstat_cutoff = float(tstat_cutoff_value)
        if not (0.0 <= tstat_cutoff <= 10.0):
            raise ValueError("tstat_cutoff must be between 0 and 10")

        kalman_q_scale = float(kalman_q_scale_value)
        if not (0.01 <= kalman_q_scale <= 10.0):
            raise ValueError("kalman_q_scale must be between 0.01 and 10.0")

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
                fm = RiskModel(
                    rf_name="Rf",
                    garch_dist=garch_dist,
                    pca_demean=False,
                    tstat_cutoff=tstat_cutoff,
                    kalman_q_scale=kalman_q_scale,
                )
                run = fm.evaluate_train_test(
                    uni=df,
                    train_fraction=train_fraction,
                    realized_window=realized_window,
                    progress=False,
                )
                results = _summarize_factor_run(run)
                run_id = _cache_run(run, meta={
                    "universe": universe,
                    "weighting": weighting,
                    "factor_set": factor_set,
                    "start_date": start_date,
                    "garch_dist": garch_dist,
                    "pca_demean": False,
                    "train_fraction": train_fraction,
                    "realized_window": realized_window,
                    "universe_data": df,
                })
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to run risk model right now.", "danger")

    confirm_overwrite = request.args.get("confirm_overwrite", "")
    if confirm_overwrite:
        run_id = request.args.get("run_id", run_id or "")

    return render_template(
        "risk_model.html",
        results=results,
        run_id=run_id,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        start_year_value=start_year_display,
        garch_dist=garch_dist,
        factor_set=factor_set,
        train_fraction_value=train_fraction_value,
        realized_window_value=realized_window_value,
        tstat_cutoff=tstat_cutoff_value,
        kalman_q_scale=kalman_q_scale_value,
        confirm_overwrite=confirm_overwrite,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )


@risk_bp.route("/risk-model/saved", methods=["GET"])
def risk_model_saved():
    selected_key = (request.args.get("run") or "").strip()
    saved_runs = []

    try:
        results_dir = _risk_results_dir()
        saved_runs = list_risk_snapshots(results_dir)

        if selected_key:
            return redirect(url_for("risk.view_saved_risk_model", run_key=selected_key))
    except FileNotFoundError:
        flash("Saved risk model not found.", "warning")
    except Exception:
        flash("Unable to load saved risk models right now.", "danger")

    return render_template(
        "risk_model_saved.html",
        saved_runs=saved_runs,
        selected_key=selected_key,
    )


@risk_bp.route("/risk-model/saved/<run_key>/view", methods=["GET"])
def view_saved_risk_model(run_key: str):
    try:
        results_dir = _risk_results_dir()
        snapshot = load_risk_snapshot(snapshot_path(results_dir, run_key))
    except FileNotFoundError:
        flash("Saved risk model not found.", "warning")
        return redirect(url_for("risk.risk_model_saved"))
    except Exception:
        flash("Unable to load the saved run right now.", "danger")
        return redirect(url_for("risk.risk_model_saved"))

    run = snapshot.run
    results = _summarize_factor_run(run)
    run_id = _cache_run(run, meta={
        "universe": snapshot.universe,
        "weighting": snapshot.weighting,
        "factor_set": snapshot.factor_set,
        "start_date": snapshot.start_date,
        "garch_dist": snapshot.garch_dist,
        "pca_demean": snapshot.pca_demean,
        "train_fraction": snapshot.train_fraction,
        "realized_window": snapshot.realized_window,
        "universe_data": snapshot.universe_data,
    })

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
        "risk_model.html",
        results=results,
        run_id=run_id,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=str(snapshot.universe),
        start_year_value=start_year_value,
        garch_dist=snapshot.garch_dist or "t",
        factor_set=snapshot.factor_set or "ff3",
        train_fraction_value=str(snapshot.train_fraction),
        realized_window_value=str(snapshot.realized_window),
        tstat_cutoff="4.0",
        confirm_overwrite="",
        earliest_start_year=earliest_start_year,
        current_year=current_year,
    )


@risk_bp.route("/risk-model/saved/<run_key>/delete", methods=["POST"])
def delete_saved_risk_model(run_key: str):
    if not request.form.get("confirm"):
        flash("Please confirm deletion before removing a saved run.", "warning")
        return redirect(url_for("risk.risk_model_saved", run=run_key))

    try:
        results_dir = _risk_results_dir()
        path = snapshot_path(results_dir, run_key)
        if not path.exists():
            flash("Saved risk model not found.", "warning")
        else:
            try:
                snap = load_risk_snapshot(path)
                display_name = snap.name
            except Exception:
                display_name = run_key
            path.unlink()
            flash(f"Deleted saved run '{display_name}'.", "success")
    except Exception:
        flash("Unable to delete the saved run right now.", "danger")

    return redirect(url_for("risk.risk_model_saved"))


@risk_bp.route("/risk-model/save", methods=["POST"])
def save_risk_model():
    run_id = request.form.get("run_id", "").strip()
    save_name = request.form.get("save_name", "").strip()
    save_overwrite = bool(request.form.get("save_overwrite"))

    if not run_id or not save_name:
        flash("Please provide a name for the risk model.", "warning")
        return redirect(url_for("risk.risk_model"))

    run = _run_cache.get(run_id)
    if run is None:
        flash("Risk model not found in cache. Please re-run the analysis.", "warning")
        return redirect(url_for("risk.risk_model"))

    meta = _run_meta_cache.get(run_id, {})

    try:
        snapshot = RiskModelSnapshot(
            name=save_name,
            created_at=datetime.now(),
            universe=meta.get("universe", 10),
            weighting=meta.get("weighting", "value"),
            factor_set=meta.get("factor_set", "ff3"),
            start_date=meta.get("start_date"),
            end_date=None,
            garch_dist=meta.get("garch_dist", "t"),
            pca_demean=meta.get("pca_demean", False),
            train_fraction=meta.get("train_fraction", 0.7),
            realized_window=meta.get("realized_window", 60),
            run=run,
            universe_data=meta.get("universe_data", pd.DataFrame()),
        )
        results_dir = Path(current_app.instance_path) / "analysis_results" / "risk_model"
        save_risk_snapshot(snapshot, results_dir, overwrite=save_overwrite)
        flash(f"Saved risk model '{save_name}'.", "success")
    except FileExistsError:
        return redirect(url_for("risk.risk_model", confirm_overwrite=save_name, run_id=run_id))
    except Exception as e:
        flash(f"Unable to save the risk model: {e}", "danger")

    return redirect(url_for("risk.risk_model"))
