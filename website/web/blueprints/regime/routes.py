"""Flask routes for the regime detection iterative builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from flask import (
    Blueprint,
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
    RegimeRun,
)
from toolkit.analysis.regime_storage import (
    RegimeCollectionInfo,
    RegimeCollectionSnapshot,
    RegimePreset,
    list_regime_collections,
    list_regime_presets,
    load_regime_collection,
    save_regime_collection,
    save_regime_preset,
)
from toolkit.analysis.style_storage import snapshot_path
from toolkit.analysis.transforms import TransformConfig, TransformType, apply_transform
from toolkit.data.fred import fetch_fred_series, search_fred_series
from toolkit.plotly_payload import summarize_regime_collection, summarize_regime_run

regime_bp = Blueprint("regime", __name__, url_prefix="/regime")


# ---------------------------------------------------------------------------
# Server-side collection cache (keyed by session ID)
# ---------------------------------------------------------------------------

_collection_cache: dict[str, RegimeCollection] = {}
_raw_series_cache: dict[str, dict[str, pd.Series]] = {}
_MAX_CACHED = 20


def _get_collection() -> RegimeCollection:
    """Get the current session's RegimeCollection."""
    sid = session.get("_regime_sid")
    if sid and sid in _collection_cache:
        return _collection_cache[sid]
    coll = RegimeCollection()
    sid = sid or _new_session_id()
    session["_regime_sid"] = sid
    _collection_cache[sid] = coll
    return coll


def _set_collection(coll: RegimeCollection) -> None:
    """Store a collection for the current session."""
    sid = session.get("_regime_sid") or _new_session_id()
    session["_regime_sid"] = sid
    if len(_collection_cache) >= _MAX_CACHED:
        oldest = next(iter(_collection_cache))
        _collection_cache.pop(oldest, None)
        _raw_series_cache.pop(oldest, None)
    _collection_cache[sid] = coll


def _get_raw_series_map() -> dict[str, pd.Series]:
    """Get the raw series cache for the current session."""
    sid = session.get("_regime_sid", "")
    return _raw_series_cache.setdefault(sid, {})


def _store_raw_series(name: str, series: pd.Series) -> None:
    """Store a raw series for the current session."""
    sid = session.get("_regime_sid", "")
    _raw_series_cache.setdefault(sid, {})[name] = series


def _remove_raw_series(name: str) -> None:
    """Remove a raw series from the current session's cache."""
    sid = session.get("_regime_sid", "")
    _raw_series_cache.get(sid, {}).pop(name, None)


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

    # Build collection summary for display
    coll_summary = None
    if collection:
        labels_map = {}
        for cfg, _ in collection.entries:
            if cfg.regime_labels:
                labels_map[cfg.name] = dict(cfg.regime_labels)
        coll_summary = summarize_regime_collection(
            collection, regime_labels_map=labels_map,
            raw_series_map=raw_map,
        )

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

    return render_template(
        "regime_detection.html",
        collection=collection,
        coll_summary=coll_summary,
        entries_table=entries_table,
        transform_types=[
            (t.value, t.name.replace("_", " ").title()) for t in TransformType
        ],
        presets=presets,
        detail_summary=detail_summary,
        detail_config=detail_config,
        view_name=view_name,
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
        train_end = (request.form.get("train_end") or "").strip() or None
        description = (request.form.get("description") or "").strip()

        config = RegimeConfig(
            name=regime_name,
            fred_series_id=fred_id,
            transform=transform,
            description=description,
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            switching_trend=switching_trend,
            train_end=train_end,
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

        # Add to collection
        collection.add(config, run)
        _set_collection(collection)

        flash(f"Fitted regime '{regime_name}' and added to collection.", "success")

    except ValueError as exc:
        flash(str(exc), "danger")
    except RuntimeError as exc:
        flash(str(exc), "danger")
    except Exception as exc:
        flash(f"Failed to fit regime: {exc}", "danger")

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
        flash(f"Removed '{name}' from collection.", "success")
    except KeyError:
        flash(f"Regime '{name}' not found.", "warning")
    return redirect(url_for("regime.regime_detection"))


# ---------------------------------------------------------------------------
# Clear collection
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/clear", methods=["POST"])
def clear_collection():
    sid = session.get("_regime_sid", "")
    _raw_series_cache.pop(sid, None)
    _set_collection(RegimeCollection())
    flash("Collection cleared.", "success")
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
        train_end = (request.form.get("train_end") or "").strip() or None

        raw_series = fetch_fred_series(fred_id)
        transformed = apply_transform(raw_series, transform)

        model = RegimeModel(
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            switching_trend=switching_trend,
        )
        run = model.run(transformed, name=fred_id, train_end=train_end)

        summary = summarize_regime_run(run, raw_series=raw_series)
        return jsonify(summary)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


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
# Save / load collection
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/save", methods=["POST"])
def save_collection_route():
    collection = _get_collection()
    if not collection:
        flash("Nothing to save — collection is empty.", "warning")
        return redirect(url_for("regime.regime_detection"))

    name = (request.form.get("collection_name") or "").strip()
    if not name:
        flash("Please provide a name for the collection.", "warning")
        return redirect(url_for("regime.regime_detection"))

    overwrite = "overwrite" in request.form

    try:
        snap = RegimeCollectionSnapshot(
            name=name,
            created_at=datetime.now(),
            collection=collection,
        )
        save_regime_collection(snap, _regime_results_dir(), overwrite=overwrite)
        flash(f"Saved collection '{name}'.", "success")
    except FileExistsError:
        flash(
            "A collection with that name already exists. Check 'Overwrite' to replace.",
            "warning",
        )
    except Exception:
        flash("Unable to save the collection right now.", "danger")

    return redirect(url_for("regime.regime_detection"))


@regime_bp.route("/regime-detection/saved", methods=["GET"])
def saved_collections():
    try:
        collections = list_regime_collections(_regime_results_dir())
    except Exception:
        collections = []
        flash("Unable to list saved collections.", "danger")

    return render_template(
        "regime_detection_saved.html",
        collections=collections,
    )


@regime_bp.route("/regime-detection/saved/<key>/load", methods=["POST"])
def load_saved_collection(key: str):
    try:
        results_dir = _regime_results_dir()
        path = snapshot_path(results_dir / "collections", key)
        snap = load_regime_collection(path)
        _set_collection(snap.collection)
        flash(f"Loaded collection '{snap.name}'.", "success")
    except FileNotFoundError:
        flash("Saved collection not found.", "warning")
    except Exception:
        flash("Unable to load the collection.", "danger")
    return redirect(url_for("regime.regime_detection"))


@regime_bp.route("/regime-detection/saved/<key>/delete", methods=["POST"])
def delete_saved_collection(key: str):
    if not request.form.get("confirm"):
        flash("Please confirm deletion.", "warning")
        return redirect(url_for("regime.saved_collections"))
    try:
        path = snapshot_path(_regime_results_dir() / "collections", key)
        if path.exists():
            path.unlink()
            flash("Deleted saved collection.", "success")
        else:
            flash("Collection not found.", "warning")
    except Exception:
        flash("Unable to delete the collection.", "danger")
    return redirect(url_for("regime.saved_collections"))


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------

@regime_bp.route("/regime-detection/save-preset", methods=["POST"])
def save_preset_route():
    collection = _get_collection()
    regime_name = (request.form.get("regime_name") or "").strip()

    if not regime_name:
        flash("Select a regime to save as preset.", "warning")
        return redirect(url_for("regime.regime_detection"))

    try:
        cfg, _ = collection.get(regime_name)
    except KeyError:
        flash(f"Regime '{regime_name}' not found.", "warning")
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
                "train_end": p.config.train_end,
            }
            for p in presets
        ])
    except Exception:
        return jsonify([])
