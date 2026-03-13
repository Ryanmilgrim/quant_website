from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

METHOD_PROJECTION = "projection"
METHOD_QP = "qp"

_DEFAULT_WINDOW_YEARS = 1.0
_STEPS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


@dataclass
class StyleRun:
    """Container for benchmark style run inputs and outputs."""

    params: dict[str, Any]
    results: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    @property
    def rolling(self) -> dict[str, Any]:
        return self.results["benchmark_style"]["rolling"]

    @property
    def weights(self) -> pd.DataFrame:
        return self.rolling["weights"]

    @property
    def tracking_weights(self) -> pd.DataFrame:
        return self.rolling["tracking_weights"]

    @property
    def alpha(self) -> pd.Series:
        return self.rolling["alpha"]

    @property
    def portfolio_return(self) -> pd.Series:
        return self.rolling["portfolio_return"]

    @property
    def benchmark_return(self) -> pd.Series:
        return self.rolling["benchmark_return"]

    @property
    def tracking_error(self) -> pd.Series:
        return self.rolling["tracking_error"]

    def summary(self) -> str:
        meta = self.results["benchmark_style"].get("meta", {})
        roll = self.rolling
        w = roll["weights"]
        pr = roll["portfolio_return"]
        window_desc = str(roll.get("window"))
        window_years = roll.get("window_years")
        window_frequency = roll.get("window_frequency")
        method = roll.get("method")
        if window_years is not None and window_frequency:
            window_desc = f"{window_desc} ({window_years:.2f} yrs, {window_frequency})"
        return (
            "StyleRun\n"
            f"  benchmark: {self.params.get('benchmark_name')}\n"
            f"  window:    {window_desc}\n"
            f"  rebalance: {roll.get('optimize_frequency')}\n"
            f"  method:    {method}\n"
            f"  assets:    {w.shape[1]}\n"
            f"  weights:   {w.index.min()} -> {w.index.max()}\n"
            f"  base:      {pr.index.min()} -> {pr.index.max()}\n"
            f"  raw_start: {meta.get('raw_start')}  raw_end: {meta.get('raw_end')}\n"
            f"  first_solve_date: {meta.get('first_solve_date')}"
        )


class StyleAnalysis:
    """
    Rolling benchmark tracking style analysis (NumPy).

    Data
    ----
    Expects `uni` from website.lib.universe.get_universe_returns(...),
    with MultiIndex columns:
      - uni["assets"]       : asset returns (X)
      - uni["benchmarks"]   : includes benchmark series (y)

    Methods
    -------
    - projection (fast approx): Solve a ridge least-squares system on centered returns, then project the
      solution onto the simplex (long-only, sum=1).
    - qp (tracking-error QP): Minimize sample tracking-error variance over the simplex using projected
      (accelerated) gradient descent.

    Important outputs
    -----------------
    - weights (rebalance dates): long-only asset weights (sum=1)
    - tracking_weights (rebalance dates): [-1, weights] (sum=0)
    - portfolio_return (base frequency): investable return X @ w (held constant between rebalances)
    - benchmark_return (base frequency): benchmark return
    - tracking_error (base frequency): portfolio_return - benchmark_return

    Note: alpha is *not* investable; it is diagnostic only.
    """

    def __init__(self, uni: pd.DataFrame, *, benchmark_name: str = "Mkt"):
        self.uni = uni
        self.benchmark_name = benchmark_name

        if not isinstance(uni.columns, pd.MultiIndex):
            raise TypeError("uni must use a MultiIndex column with 'assets' and 'benchmarks' groups.")
        if "assets" not in uni.columns.get_level_values(0):
            raise KeyError("uni must contain top-level group 'assets'")
        if "benchmarks" not in uni.columns.get_level_values(0):
            raise KeyError("uni must contain top-level group 'benchmarks'")
        if benchmark_name not in uni["benchmarks"].columns:
            raise KeyError(f"Benchmark '{benchmark_name}' not found in uni['benchmarks']")
        if uni["assets"].shape[1] == 0:
            raise ValueError("uni['assets'] has no columns.")

    def run(
        self,
        *,
        style_window: Optional[int] = None,
        style_window_years: Optional[float] = None,
        optimize_frequency: str = "daily",  # "daily" | "weekly" | "monthly" | "annual"
        method: str = METHOD_PROJECTION,  # "projection" | "qp"
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> StyleRun:
        assets, bench = self._split_universe(start=start, end=end)

        style = self._run_benchmark_style(
            benchmark=bench,
            assets=assets,
            window=style_window,
            window_years=style_window_years,
            optimize_frequency=optimize_frequency,
            method=method,
        )

        params = {
            "benchmark_name": self.benchmark_name,
            "style_window": style["rolling"]["window"],
            "style_window_years": style["rolling"]["window_years"],
            "window_frequency": style["rolling"]["window_frequency"],
            "optimize_frequency": style["rolling"]["optimize_frequency"],
            "method": style["rolling"]["method"],
            "start": start,
            "end": end,
        }
        return StyleRun(params=params, results={"benchmark_style": style})

    def _split_universe(
        self,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assets = self.uni["assets"].copy().sort_index()
        bench = self.uni["benchmarks"][self.benchmark_name].copy().sort_index()

        idx = assets.index.intersection(bench.index)
        assets = assets.loc[idx]
        bench = bench.loc[idx]

        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        if start_ts is not None:
            assets = assets.loc[assets.index >= start_ts]
            bench = bench.loc[bench.index >= start_ts]
        if end_ts is not None:
            assets = assets.loc[assets.index < end_ts]
            bench = bench.loc[bench.index < end_ts]

        assets = assets.replace([np.inf, -np.inf], np.nan)
        bench = bench.replace([np.inf, -np.inf], np.nan)

        keep = bench.notna()
        assets = assets.loc[keep].astype(float)
        bench = bench.loc[keep].astype(float)

        if assets.empty or bench.empty:
            raise ValueError("No data available after applying filters.")

        return assets, bench

    def _run_benchmark_style(
        self,
        *,
        benchmark: pd.Series,
        assets: pd.DataFrame,
        window: Optional[int],
        window_years: Optional[float],
        optimize_frequency: str,
        method: str,
    ) -> dict[str, Any]:
        df0 = pd.concat([benchmark.rename(self.benchmark_name), assets], axis=1).sort_index()
        raw_start = df0.index.min()
        raw_end = df0.index.max()

        df = df0[[self.benchmark_name] + [c for c in df0.columns if c != self.benchmark_name]]

        frequency = _infer_frequency(df.index)
        steps_per_year = _steps_per_year(frequency)

        if window is None:
            years = float(_DEFAULT_WINDOW_YEARS if window_years is None else window_years)
            if years <= 0:
                raise ValueError("style_window_years must be positive.")
            window = int(round(years * steps_per_year))
        else:
            window = int(window)
            years = window / float(steps_per_year)

        if window <= 1:
            raise ValueError("style_window must be greater than 1.")
        if len(df) < window:
            raise ValueError(f"Not enough rows ({len(df)}) for window={window}")

        rebalance = _normalize_rebalance(optimize_frequency)
        solve_method = _normalize_method(method)

        out = _rolling_tracking(
            df=df,
            window=window,
            optimize_frequency=rebalance,
            method=solve_method,
        )

        meta = {
            "raw_start": raw_start,
            "raw_end": raw_end,
            "first_solve_date": out["weights"].index.min() if len(out["weights"]) else None,
        }

        return {
            "meta": meta,
            "rolling": {
                "window": window,
                "window_years": years,
                "window_frequency": frequency,
                "optimize_frequency": rebalance,
                "method": solve_method,
                "weights": out["weights"],
                "tracking_weights": out["tracking_weights"],
                "alpha": out["alpha"],  # diagnostics only (not investable)
                "portfolio_return": out["portfolio_return"],
                "benchmark_return": out["benchmark_return"],
                "tracking_error": out["tracking_error"],
            },
        }


def _rolling_tracking(
    *,
    df: pd.DataFrame,
    window: int,
    optimize_frequency: str,
    method: str,
) -> dict[str, Any]:
    """
    Solve at rebalance dates; hold weights constant between rebalances.
    Base-frequency data is always used inside each window.

    Missing assets:
      - for each window, any asset with a NaN inside the window is forced to weight 0.
    """
    df = df.sort_index()
    Z = df.to_numpy(dtype=float)  # benchmark + assets (assets may include NaN)
    dates = pd.DatetimeIndex(df.index)
    cols = list(df.columns)
    bmk_col = cols[0]
    asset_cols = cols[1:]

    T, n_plus_1 = Z.shape
    n_assets = n_plus_1 - 1
    if n_assets <= 0:
        raise ValueError("At least one asset series is required.")

    reb_dates = _rebalance_dates(dates, optimize_frequency)
    first_usable = dates[window - 1]
    reb_dates = reb_dates[reb_dates >= first_usable]

    reb_pos = dates.get_indexer(reb_dates)
    reb_pos = reb_pos[(reb_pos >= (window - 1)) & (reb_pos >= 0)]

    solve_at = np.zeros(T, dtype=bool)
    solve_at[reb_pos] = True

    y = Z[:, 0]
    X = Z[:, 1:]
    y_filled = np.nan_to_num(y, nan=0.0)
    nan_mask = np.isnan(X)
    has_nans = bool(nan_mask.any())
    X_filled = np.nan_to_num(X, nan=0.0)

    # Rolling state for window sums.
    sx = X_filled[:window].sum(axis=0)
    sy = float(y_filled[:window].sum())
    Xty = X_filled[:window].T @ y_filled[:window]
    XtX = X_filled[:window].T @ X_filled[:window]
    nan_counts = nan_mask[:window].sum(axis=0).astype(np.int64) if has_nans else np.zeros(n_assets, dtype=np.int64)

    solved_pos: list[int] = []
    solved_w: list[np.ndarray] = []
    solved_a: list[float] = []
    prev_w: Optional[np.ndarray] = None

    inv_window = 1.0 / float(window)

    for t_end in range(window - 1, T):
        if solve_at[t_end]:
            avail = nan_counts == 0
            if avail.any():
                G = XtX - np.outer(sx, sx) * inv_window
                g = Xty - sx * (sy * inv_window)

                if avail.all():
                    G_avail = G
                    g_avail = g
                    w0 = prev_w
                else:
                    idx = np.flatnonzero(avail)
                    G_avail = G[np.ix_(idx, idx)]
                    g_avail = g[idx]
                    w0 = prev_w[idx] if prev_w is not None else None

                if w0 is not None:
                    s0 = float(np.sum(w0))
                    w0 = (w0 / s0) if s0 > 0 else None

                if method == METHOD_PROJECTION:
                    w_avail = _solve_projection(G_avail, g_avail)
                else:
                    w_avail = _solve_qp(G_avail, g_avail, w0=w0)

                w_full = np.zeros(n_assets, dtype=float)
                w_full[avail] = w_avail

                mu_x = sx * inv_window
                mu_y = sy * inv_window
                alpha = float(mu_y - float(mu_x @ w_full))

                solved_pos.append(int(t_end))
                solved_w.append(w_full)
                solved_a.append(alpha)
                prev_w = w_full

        if (t_end + 1) >= T:
            break

        s = t_end - window + 1
        add = t_end + 1

        x_out = X_filled[s]
        x_in = X_filled[add]
        y_out = float(y_filled[s])
        y_in = float(y_filled[add])

        sx = sx + x_in - x_out
        sy = sy + y_in - y_out
        Xty = Xty + (x_in * y_in) - (x_out * y_out)
        XtX = XtX + np.outer(x_in, x_in) - np.outer(x_out, x_out)

        if has_nans:
            nan_counts = nan_counts + nan_mask[add].astype(np.int64) - nan_mask[s].astype(np.int64)

    if len(solved_pos) == 0:
        empty_idx = pd.DatetimeIndex([])
        empty_w = pd.DataFrame(index=empty_idx, columns=asset_cols, dtype=float)
        empty_tw = pd.DataFrame(index=empty_idx, columns=[bmk_col] + asset_cols, dtype=float)
        base_idx = dates
        bench_s = pd.Series(y, index=base_idx, name="benchmark_return")
        nan_s = pd.Series(np.full(T, np.nan), index=base_idx, name="portfolio_return")
        return {
            "weights": empty_w,
            "tracking_weights": empty_tw,
            "alpha": pd.Series(dtype=float),
            "portfolio_return": nan_s,
            "benchmark_return": bench_s,
            "tracking_error": (nan_s - bench_s).rename("tracking_error"),
        }

    solved_dates = dates[solved_pos]

    W = pd.DataFrame(np.vstack(solved_w), index=solved_dates, columns=asset_cols)
    TW = pd.DataFrame(
        np.column_stack([np.full(len(solved_dates), -1.0), W.to_numpy()]),
        index=solved_dates,
        columns=[bmk_col] + asset_cols,
    )
    A = pd.Series(solved_a, index=solved_dates, name="alpha")

    # base-frequency investable return with piecewise-constant weights
    portfolio_ret = np.full(T, np.nan, dtype=float)
    for i, pos in enumerate(solved_pos):
        seg_start = pos
        seg_end = solved_pos[i + 1] if (i + 1) < len(solved_pos) else T
        portfolio_ret[seg_start:seg_end] = X_filled[seg_start:seg_end, :] @ solved_w[i]

    bench_s = pd.Series(y, index=dates, name="benchmark_return")
    port_s = pd.Series(portfolio_ret, index=dates, name="portfolio_return")
    te_s = (port_s - bench_s).rename("tracking_error")

    return {
        "weights": W,
        "tracking_weights": TW,
        "alpha": A,
        "portfolio_return": port_s,
        "benchmark_return": bench_s,
        "tracking_error": te_s,
    }


def _solve_projection(G: np.ndarray, g: np.ndarray, *, ridge_rel: float = 1e-6) -> np.ndarray:
    n = int(g.shape[0])
    if n <= 0:
        raise ValueError("Empty solve.")
    if n == 1:
        return np.array([1.0], dtype=float)

    Gs = 0.5 * (G + G.T)
    avg_diag = float(np.trace(Gs)) / float(n)
    ridge = ridge_rel * max(avg_diag, 1e-12)

    try:
        w = np.linalg.solve(Gs + ridge * np.eye(n), g)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Gs + ridge * np.eye(n), g, rcond=None)[0]

    if not np.isfinite(w).all():
        w = np.zeros(n, dtype=float)
        j = int(np.nanargmax(g)) if np.isfinite(g).any() else 0
        w[j] = 1.0
        return w

    return _project_to_simplex(w)


def _solve_qp(
    G: np.ndarray,
    g: np.ndarray,
    *,
    w0: Optional[np.ndarray],
    ridge_rel: float = 1e-6,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    n = int(g.shape[0])
    if n <= 0:
        raise ValueError("Empty solve.")
    if n == 1:
        return np.array([1.0], dtype=float)

    Gs = 0.5 * (G + G.T)
    avg_diag = float(np.trace(Gs)) / float(n)
    ridge = ridge_rel * max(avg_diag, 1e-12)
    Gs = Gs + ridge * np.eye(n)

    try:
        L = float(np.linalg.eigvalsh(Gs).max())
    except np.linalg.LinAlgError:
        L = float(np.linalg.norm(Gs, ord=2))

    if not np.isfinite(L) or L <= 0:
        return _solve_projection(G, g, ridge_rel=ridge_rel)

    step = 1.0 / L

    if w0 is None or w0.shape != (n,) or not np.isfinite(w0).all():
        w = np.full(n, 1.0 / float(n), dtype=float)
    else:
        w = _project_to_simplex(w0)

    z = w.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = (Gs @ z) - g
        w_next = _project_to_simplex(z - step * grad)
        if float(np.max(np.abs(w_next - w))) <= tol:
            w = w_next
            break
        t_next = (1.0 + float(np.sqrt(1.0 + 4.0 * t * t))) * 0.5
        z = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w = w_next
        t = t_next

    return w


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex (w>=0, sum(w)=1)."""
    x = np.asarray(v, dtype=float).reshape(-1)
    n = x.size
    if n == 1:
        return np.array([1.0], dtype=float)

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - (cssv / ind) > 0
    if not np.any(cond):
        return np.full(n, 1.0 / float(n), dtype=float)

    rho = int(np.nonzero(cond)[0][-1])
    theta = float(cssv[rho] / float(rho + 1))
    w = np.maximum(x - theta, 0.0)

    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        return np.full(n, 1.0 / float(n), dtype=float)
    if abs(s - 1.0) > 1e-12:
        w = w / s
    return w


def _infer_frequency(idx: pd.Index) -> str:
    if len(idx) < 3:
        return "daily"

    inferred = pd.infer_freq(pd.DatetimeIndex(idx))
    if inferred:
        if inferred.startswith(("B", "D")):
            return "daily"
        if inferred.startswith("W"):
            return "weekly"
        if inferred.startswith(("M", "MS")):
            return "monthly"

    dt = pd.to_datetime(idx)
    deltas = np.diff(dt.values.astype("datetime64[D]").astype(np.int64))
    med = float(np.median(deltas)) if len(deltas) else 1.0
    if med <= 3:
        return "daily"
    if med <= 10:
        return "weekly"
    return "monthly"


def _steps_per_year(frequency: str) -> int:
    return _STEPS_PER_YEAR.get(frequency, 252)


def _normalize_rebalance(freq: str) -> str:
    f = (freq or "daily").strip().lower()
    if f in ("d", "day", "daily"):
        return "daily"
    if f in ("w", "week", "weekly"):
        return "weekly"
    if f in ("m", "month", "monthly"):
        return "monthly"
    if f in ("a", "y", "yr", "year", "yearly", "annual", "annually"):
        return "annual"
    raise ValueError("optimize_frequency must be 'daily', 'weekly', 'monthly', or 'annual'")


def _normalize_method(method: str) -> str:
    m = (method or METHOD_PROJECTION).strip().lower()
    if m in (METHOD_PROJECTION, "ls", "least_squares", "least-squares"):
        return METHOD_PROJECTION
    if m in (METHOD_QP, "te", "tracking_error", "tracking-error"):
        return METHOD_QP
    raise ValueError("method must be 'projection' or 'qp'")


def _rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "daily":
        return idx

    s = idx.to_series(index=idx)

    if freq == "weekly":
        # last available base date of each (Fri-ended) week
        d = s.groupby(pd.Grouper(freq="W-FRI")).max().dropna().sort_values().values
        return pd.DatetimeIndex(d)

    if freq == "annual":
        d = s.groupby(pd.Grouper(freq="YE")).max().dropna().sort_values().values
        return pd.DatetimeIndex(d)

    # monthly: last available base date of each month
    d = s.groupby(pd.Grouper(freq="ME")).max().dropna().sort_values().values
    return pd.DatetimeIndex(d)


__all__ = [
    "METHOD_PROJECTION",
    "METHOD_QP",
    "StyleAnalysis",
    "StyleRun",
]
