"""Flask routes for the regime detection iterative builder."""

from __future__ import annotations

import io
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from toolkit.analysis.regime_detection import (
    RegimeCollection,
    RegimeConfig,
    RegimeModel,
)
from toolkit.analysis.regime_storage import (
    RegimeCollectionSnapshot,
    RegimePreset,
    list_regime_presets,
    load_regime_collection,
    save_regime_collection,
    save_regime_preset,
)
from toolkit.analysis.transforms import TransformConfig, TransformType, apply_transform
from toolkit.data.fred import fetch_fred_series, search_fred_series
from toolkit.plotly_payload import summarize_regime_run

regime_bp = Blueprint("regime", __name__, url_prefix="/regime")


# ---------------------------------------------------------------------------
# Server-side collection cache (keyed by session ID)
# ---------------------------------------------------------------------------

_collection_cache: dict[str, RegimeCollection] = {}
_raw_series_cache: dict[str, dict[str, pd.Series]] = {}
_MAX_CACHED = 20


def _active_pkl_path() -> Path:
    """Path to the auto-saved active collection pickle."""
    return _regime_results_dir() / "collections" / "_active.pkl"


def _active_raw_pkl_path() -> Path:
    """Path to the auto-saved raw series pickle."""
    return _regime_results_dir() / "collections" / "_active_raw.pkl"


def _auto_save(collection: RegimeCollection) -> None:
    """Persist the active collection to disk automatically."""
    try:
        snap = RegimeCollectionSnapshot(
            name="_active",
            created_at=datetime.now(),
            collection=collection,
        )
        save_regime_collection(snap, _regime_results_dir(), overwrite=True)
    except Exception:
        pass


def _auto_save_raw() -> None:
    """Persist the raw series cache to disk."""
    sid = session.get("_regime_sid", "")
    raw_map = _raw_series_cache.get(sid, {})
    try:
        path = _active_raw_pkl_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(raw_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _get_collection() -> RegimeCollection:
    """Get the current session's RegimeCollection, auto-loading from disk if needed."""
    sid = session.get("_regime_sid")
    if sid and sid in _collection_cache:
        return _collection_cache[sid]
    # Try to load auto-saved collection from disk
    try:
        path = _active_pkl_path()
        if path.exists():
            snap = load_regime_collection(path)
            sid = sid or _new_session_id()
            session["_regime_sid"] = sid
            _collection_cache[sid] = snap.collection
            return snap.collection
    except Exception:
        pass
    coll = RegimeCollection()
    sid = sid or _new_session_id()
    session["_regime_sid"] = sid
    _collection_cache[sid] = coll
    return coll


def _set_collection(coll: RegimeCollection) -> None:
    """Store a collection for the current session and auto-save to disk."""
    sid = session.get("_regime_sid") or _new_session_id()
    session["_regime_sid"] = sid
    if len(_collection_cache) >= _MAX_CACHED:
        oldest = next(iter(_collection_cache))
        _collection_cache.pop(oldest, None)
        _raw_series_cache.pop(oldest, None)
    _collection_cache[sid] = coll
    _auto_save(coll)


def _get_raw_series_map() -> dict[str, pd.Series]:
    """Get the raw series cache, auto-loading from disk if needed."""
    sid = session.get("_regime_sid", "")
    if sid in _raw_series_cache and _raw_series_cache[sid]:
        return _raw_series_cache[sid]
    # Try to load from disk
    try:
        path = _active_raw_pkl_path()
        if path.exists():
            with path.open("rb") as f:
                raw_map = pickle.load(f)
            _raw_series_cache[sid] = raw_map
            return raw_map
    except Exception:
        pass
    return _raw_series_cache.setdefault(sid, {})


def _store_raw_series(name: str, series: pd.Series) -> None:
    """Store a raw series for the current session and auto-save."""
    sid = session.get("_regime_sid", "")
    _raw_series_cache.setdefault(sid, {})[name] = series
    _auto_save_raw()


def _remove_raw_series(name: str) -> None:
    """Remove a raw series from the current session's cache and auto-save."""
    sid = session.get("_regime_sid", "")
    _raw_series_cache.get(sid, {}).pop(name, None)
    _auto_save_raw()


def _new_session_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _infer_frequency(series: pd.Series) -> str:
    """Infer the frequency label from a series' DatetimeIndex."""
    try:
        freq = pd.infer_freq(series.index)
        if freq:
            freq_map = {
                "D": "Daily", "B": "Business daily",
                "W": "Weekly", "MS": "Monthly", "ME": "Monthly",
                "M": "Monthly", "QS": "Quarterly", "QE": "Quarterly",
                "Q": "Quarterly", "YS": "Annual", "YE": "Annual",
                "A": "Annual", "AS": "Annual",
            }
            # Handle prefixed frequencies like Q-DEC, QS-OCT
            base = freq.split("-")[0] if "-" in freq else freq
            return freq_map.get(base, freq)
    except Exception:
        pass
    # Fallback: estimate from median gap between observations
    if len(series) >= 2:
        median_days = float(
            np.median(np.diff(series.index).astype("timedelta64[D]").astype(int))
        )
        if median_days <= 3:
            return "Daily"
        if median_days <= 10:
            return "Weekly"
        if median_days <= 45:
            return "Monthly"
        if median_days <= 120:
            return "Quarterly"
        return "Annual"
    return "Unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_results_dir() -> Path:
    root = Path(current_app.instance_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_results" / "regime_detection"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_transform_type(val: str) -> TransformType:
    try:
        return TransformType(val)
    except ValueError:
        return TransformType.NONE


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection", methods=["GET"])
def regime_detection():
    collection = _get_collection()
    raw_map = _get_raw_series_map()

    # Build collection entries table data
    entries_table = []
    for cfg, run in collection.entries:
        m = run.meta
        raw_s = raw_map.get(cfg.name)
        n_total = len(raw_s) if raw_s is not None else m.get("n_obs", "?")
        n_missing = int(raw_s.isna().sum()) if raw_s is not None else 0
        freq = _infer_frequency(raw_s) if raw_s is not None else "?"
        entries_table.append({
            "name": cfg.name,
            "fred_series_id": cfg.fred_series_id,
            "k_regimes": cfg.k_regimes,
            "n_obs": m.get("n_obs", "?"),
            "start_date": m.get("start_date", "?"),
            "frequency": freq,
            "missing_obs": n_missing,
            "description": cfg.description,
            # For collapsible transform details
            "transform": cfg.transform.description,
            "train_end": cfg.train_end or "full",
            "switching_variance": cfg.switching_variance,
            "switching_trend": cfg.switching_trend,
            "order": cfg.order,
            "switching_ar": cfg.switching_ar,
            "converged": m.get("converged", True),
        })

    # Load saved presets for the form dropdown
    try:
        presets = list_regime_presets(_regime_results_dir())
    except Exception:
        presets = []

    # Single-regime detail if requested via query param
    view_name = request.args.get("view")
    detail_summary = None
    detail_config = None
    if view_name:
        try:
            cfg, run = collection.get(view_name)
            labels = dict(cfg.regime_labels) if cfg.regime_labels else None
            raw_s = raw_map.get(view_name)
            detail_summary = summarize_regime_run(
                run, regime_labels=labels, raw_series=raw_s,
            )
            detail_config = cfg
        except KeyError:
            pass

    # Serialize config for JS pre-populate when viewing a saved regime
    view_config_json = None
    if detail_config:
        view_config_json = {
            "fred_series_id": detail_config.fred_series_id,
            "transform_type": detail_config.transform.transform.value,
            "transform_window": detail_config.transform.window,
            "k_regimes": detail_config.k_regimes,
            "switching_variance": detail_config.switching_variance,
            "switching_trend": detail_config.switching_trend,
            "order": detail_config.order,
            "switching_ar": detail_config.switching_ar,
            "train_end": detail_config.train_end or "",
            "description": detail_config.description or "",
            "name": detail_config.name,
        }

    # Default training cutoff = most recent Dec 31
    today = date.today()
    default_train_end = date(today.year - 1, 12, 31).isoformat()

    return render_template(
        "regime_detection.html",
        collection=collection,
        entries_table=entries_table,
        transform_types=[
            (t.value, t.name.replace("_", " ").title()) for t in TransformType
        ],
        presets=presets,
        detail_summary=detail_summary,
        detail_config=detail_config,
        view_name=view_name,
        view_config_json=view_config_json,
        default_train_end=default_train_end,
    )


# ---------------------------------------------------------------------------
# Fit regime (add to collection)
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/fit", methods=["POST"])
def fit_regime():
    collection = _get_collection()

    try:
        regime_name = (request.form.get("regime_name") or "").strip()
        if not regime_name:
            raise ValueError("Regime name is required.")

        fred_id = (request.form.get("fred_series_id") or "").strip().upper()
        if not fred_id:
            raise ValueError("FRED series ID is required.")

        transform_type = _parse_transform_type(
            request.form.get("transform_type", "none")
        )
        window = int(request.form.get("transform_window", "4") or "4")
        transform = TransformConfig(transform=transform_type, window=window)

        k_regimes = int(request.form.get("k_regimes", "2"))
        switching_variance = "switching_variance" in request.form
        switching_trend = "switching_trend" in request.form
        order = int(request.form.get("order", "0") or "0")
        switching_ar = "switching_ar" in request.form
        train_end = (request.form.get("train_end") or "").strip() or None
        description = (request.form.get("description") or "").strip()

        # Collect regime labels from preview inputs (label_0, label_1, ...)
        regime_labels = {
            i: lbl
            for i in range(k_regimes)
            if (lbl := (request.form.get(f"label_{i}") or "").strip())
        } or None

        config = RegimeConfig(
            name=regime_name,
            fred_series_id=fred_id,
            transform=transform,
            description=description,
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            switching_trend=switching_trend,
            order=order,
            switching_ar=switching_ar,
            train_end=train_end,
            regime_labels=regime_labels,
        )

        # Fetch FRED data
        raw_series = fetch_fred_series(fred_id)

        # Apply transform
        transformed = apply_transform(raw_series, transform)

        # Fit model
        model = RegimeModel.from_config(config)
        run = model.run(transformed, name=regime_name, train_end=train_end)

        # Store raw series for dual-axis charts
        _store_raw_series(regime_name, raw_series)

        # Remove existing regime with same name (overwrite)
        try:
            collection.remove(regime_name)
        except KeyError:
            pass
        collection.add(config, run)
        _set_collection(collection)

        flash(f"Fitted signal '{regime_name}' and added to data lake.", "success")
        return redirect(url_for("regime.regime_detection"))

    except ValueError as exc:
        flash(str(exc), "danger")
    except RuntimeError as exc:
        msg = str(exc)
        if "converge" in msg.lower():
            msg += " Try submitting again, or reduce the number of regimes."
        flash(msg, "danger")
    except Exception as exc:
        flash(f"Failed to fit signal: {exc}", "danger")

    return redirect(url_for("regime.regime_detection"))


# Keep old URL as alias for backwards compatibility
@regime_bp.route("/regime-detection/add", methods=["POST"])
def add_regime():
    return fit_regime()


# ---------------------------------------------------------------------------
# Remove regime from collection
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/remove/<name>", methods=["POST"])
def remove_regime(name: str):
    collection = _get_collection()
    try:
        collection.remove(name)
        _remove_raw_series(name)
        _set_collection(collection)
        flash(f"Removed '{name}' from data lake.", "success")
    except KeyError:
        flash(f"Signal '{name}' not found.", "warning")
    return redirect(url_for("regime.regime_detection"))


# ---------------------------------------------------------------------------
# Refit an existing regime with updated model settings
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/refit/<name>", methods=["POST"])
def refit_regime(name: str):
    collection = _get_collection()
    try:
        cfg_old, _ = collection.get(name)

        k_regimes = int(request.form.get("k_regimes", cfg_old.k_regimes))
        switching_variance = "switching_variance" in request.form
        switching_trend = "switching_trend" in request.form
        order = int(request.form.get("order", cfg_old.order) or "0")
        switching_ar = "switching_ar" in request.form
        train_end = (request.form.get("train_end") or "").strip() or None

        config = RegimeConfig(
            name=name,
            fred_series_id=cfg_old.fred_series_id,
            transform=cfg_old.transform,
            description=cfg_old.description,
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            switching_trend=switching_trend,
            order=order,
            switching_ar=switching_ar,
            train_end=train_end,
        )

        raw_series = fetch_fred_series(cfg_old.fred_series_id)
        transformed = apply_transform(raw_series, cfg_old.transform)
        model = RegimeModel.from_config(config)
        run = model.run(transformed, name=name, train_end=train_end)

        _store_raw_series(name, raw_series)
        collection.remove(name)
        collection.add(config, run)
        _set_collection(collection)

        flash(f"Refitted '{name}' and updated in data lake.", "success")
        return redirect(url_for("regime.regime_detection", view=name))

    except KeyError:
        flash(f"Signal '{name}' not found.", "warning")
    except ValueError as exc:
        flash(str(exc), "danger")
    except RuntimeError as exc:
        msg = str(exc)
        if "converge" in msg.lower():
            msg += " Try submitting again, or reduce the number of regimes."
        flash(msg, "danger")
    except Exception as exc:
        flash(f"Failed to refit signal: {exc}", "danger")

    return redirect(url_for("regime.regime_detection", view=name))


# ---------------------------------------------------------------------------
# Update regime labels (no refit)
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/label/<name>", methods=["POST"])
def label_regime(name: str):
    collection = _get_collection()
    try:
        cfg_old, run = collection.get(name)
        labels = {
            i: lbl
            for i in range(cfg_old.k_regimes)
            if (lbl := (request.form.get(f"label_{i}") or "").strip())
        }
        config = RegimeConfig(
            name=cfg_old.name,
            fred_series_id=cfg_old.fred_series_id,
            transform=cfg_old.transform,
            description=cfg_old.description,
            k_regimes=cfg_old.k_regimes,
            switching_variance=cfg_old.switching_variance,
            switching_trend=cfg_old.switching_trend,
            train_end=cfg_old.train_end,
            regime_labels=labels if labels else None,
        )
        collection.remove(name)
        collection.add(config, run)
        _set_collection(collection)
        flash(f"Labels updated for '{name}'.", "success")
    except KeyError:
        flash(f"Signal '{name}' not found.", "warning")
    except Exception as exc:
        flash(f"Failed to update labels: {exc}", "danger")
    return redirect(url_for("regime.regime_detection", view=name))


# ---------------------------------------------------------------------------
# Clear collection
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/clear", methods=["POST"])
def clear_collection():
    sid = session.get("_regime_sid", "")
    _raw_series_cache.pop(sid, None)
    _set_collection(RegimeCollection())
    _auto_save_raw()
    flash("Data lake cleared.", "success")
    return redirect(url_for("regime.regime_detection"))


# ---------------------------------------------------------------------------
# Preview a single regime (AJAX)
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/preview", methods=["POST"])
def preview_regime():
    try:
        fred_id = (request.form.get("fred_series_id") or "").strip().upper()
        if not fred_id:
            return jsonify({"error": "FRED series ID is required."}), 400

        transform_type = _parse_transform_type(
            request.form.get("transform_type", "none")
        )
        window = int(request.form.get("transform_window", "4") or "4")
        transform = TransformConfig(transform=transform_type, window=window)

        k_regimes = int(request.form.get("k_regimes", "2"))
        switching_variance = request.form.get("switching_variance") == "true"
        switching_trend = request.form.get("switching_trend") == "true"
        order = int(request.form.get("order", "0") or "0")
        switching_ar = request.form.get("switching_ar") == "true"
        train_end = (request.form.get("train_end") or "").strip() or None

        raw_series = fetch_fred_series(fred_id)
        transformed = apply_transform(raw_series, transform)

        model = RegimeModel(
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            switching_trend=switching_trend,
            order=order,
            switching_ar=switching_ar,
        )
        run = model.run(transformed, name=fred_id, train_end=train_end)

        summary = summarize_regime_run(run, raw_series=raw_series)
        return jsonify(summary)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Inspect series without fitting (AJAX)
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/inspect-data", methods=["POST"])
def inspect_data():
    try:
        fred_id = (request.form.get("fred_series_id") or "").strip().upper()
        if not fred_id:
            return jsonify({"error": "FRED series ID is required."}), 400
        transform_type = _parse_transform_type(request.form.get("transform_type", "none"))
        window = int(request.form.get("transform_window", "4") or "4")
        transform = TransformConfig(transform=transform_type, window=window)

        raw = fetch_fred_series(fred_id)
        transformed = apply_transform(raw, transform)

        from toolkit.plotly_payload import line_chart_payload
        chart_series = line_chart_payload(
            transformed.rename(fred_id).to_frame(), y_axis_title=fred_id
        )

        metadata = {
            "n_obs": len(transformed),
            "frequency": _infer_frequency(transformed),
            "start": transformed.index.min().date().isoformat(),
            "end": transformed.index.max().date().isoformat(),
            "missing": int(raw.isna().sum()),
            "mean": round(float(transformed.mean()), 4),
            "std": round(float(transformed.std()), 4),
            "min": round(float(transformed.min()), 4),
            "max": round(float(transformed.max()), 4),
        }
        return jsonify({
            "chart_series": chart_series,
            "metadata": metadata,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# CSV download
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/download/<name>", methods=["GET"])
def download_regime_csv(name: str):
    """Download soft regime probabilities as a business-day CSV."""
    collection = _get_collection()
    try:
        cfg, run = collection.get(name)
    except KeyError:
        flash(f"Signal '{name}' not found.", "danger")
        return redirect(url_for("regime.regime_detection"))

    probs = run.smoothed_probabilities.copy()

    # Rename columns using labels if available
    if cfg.regime_labels:
        probs.columns = [
            cfg.regime_labels.get(i, col)
            for i, col in enumerate(probs.columns)
        ]

    # Resample to business day frequency, forward-fill (no data leakage)
    probs = probs.resample("B").ffill()

    buf = io.StringIO()
    probs.to_csv(buf, index=True, index_label="date")
    resp = Response(buf.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = (
        f'attachment; filename="{name}_regime_probs.csv"'
    )
    return resp


# ---------------------------------------------------------------------------
# FRED search (AJAX)
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/search-fred", methods=["GET"])
def search_fred():
    query = (request.args.get("q") or "").strip()
    if len(query) < 2:
        return jsonify([])
    try:
        results = search_fred_series(query, limit=10)
        return jsonify(results)
    except Exception:
        return jsonify([])


# ---------------------------------------------------------------------------
# Data lake management page
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/saved", methods=["GET"])
def saved_collections():
    collection = _get_collection()
    raw_map = _get_raw_series_map()

    entries_table = []
    for cfg, run in collection.entries:
        m = run.meta
        raw_s = raw_map.get(cfg.name)
        n_missing = int(raw_s.isna().sum()) if raw_s is not None else 0
        freq = _infer_frequency(raw_s) if raw_s is not None else "?"
        entries_table.append({
            "name": cfg.name,
            "fred_series_id": cfg.fred_series_id,
            "k_regimes": cfg.k_regimes,
            "n_obs": m.get("n_obs", "?"),
            "start_date": m.get("start_date", "?"),
            "frequency": freq,
            "missing_obs": n_missing,
            "converged": m.get("converged", True),
        })

    return render_template(
        "regime_detection_saved.html",
        entries_table=entries_table,
    )


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/save-preset", methods=["POST"])
def save_preset_route():
    collection = _get_collection()
    regime_name = (request.form.get("regime_name") or "").strip()

    if not regime_name:
        flash("Select a signal to save as preset.", "warning")
        return redirect(url_for("regime.regime_detection"))

    try:
        cfg, _ = collection.get(regime_name)
    except KeyError:
        flash(f"Signal '{regime_name}' not found.", "warning")
        return redirect(url_for("regime.regime_detection"))

    preset_name = (request.form.get("preset_name") or regime_name).strip()
    preset_desc = (request.form.get("preset_description") or "").strip()

    try:
        preset = RegimePreset(
            name=preset_name,
            description=preset_desc,
            config=cfg,
            created_at=datetime.now(),
        )
        save_regime_preset(preset, _regime_results_dir(), overwrite=True)
        flash(f"Saved preset '{preset_name}'.", "success")
    except Exception:
        flash("Unable to save preset.", "danger")

    return redirect(url_for("regime.regime_detection"))


@regime_bp.route("/regime-detection/presets", methods=["GET"])
def list_presets_json():
    """AJAX endpoint returning saved presets as JSON."""
    try:
        presets = list_regime_presets(_regime_results_dir())
        return jsonify([
            {
                "name": p.name,
                "description": p.description,
                "fred_series_id": p.config.fred_series_id,
                "transform": p.config.transform.transform.value,
                "transform_window": p.config.transform.window,
                "k_regimes": p.config.k_regimes,
                "switching_variance": p.config.switching_variance,
                "switching_trend": p.config.switching_trend,
                "order": p.config.order,
                "switching_ar": p.config.switching_ar,
                "train_end": p.config.train_end,
            }
            for p in presets
        ])
    except Exception:
        return jsonify([])
