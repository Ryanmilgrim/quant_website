# TODO -- Quant Dashboard Roadmap

## Recently Completed

- [x] **MarkovAutoregression** -- AR Order and Switching AR support in regime signals (Hamilton 1989)
- [x] **Regime signal polish** -- Annualized stats, NaN-safe JSON, regime overlay legend, param name remapping after reorder, AIC/BIC near Fit button, frequency inference fallback, "regime signal" UI rename, AR controls layout
- [x] **Rename: Factor Analysis -> Risk Model** -- Blueprint, routes, templates, nav, homepage cards all updated
- [x] **Theme & Styling Cleanup** -- CSS custom properties defined; hardcoded colors migrated to variables; info-tip tooltips and chart-source citation classes added
- [x] **Descriptions & Transparency** -- Homepage rewritten with richer module descriptions; collapsible Model Overview sections added to all modules; tooltips on form labels; chart source citations; Plotly theme centralized in base.html
- [x] **Section numbering fix** -- Regime Signals corrected to 05; Risk Model saved view corrected to 06; nav order matches numbering

## Bugs / Known Issues

- [ ] **Regression summary mismatch on view load** -- When viewing a saved signal from the data lake, the regression summary (significant variables table) does not render identically to how it appears on the builder page after fitting. Investigate why the view-mode path produces different output than the fit-preview path.

## Cleanup (minor)

- [ ] Delete empty `website/web/blueprints/factor/` directory
- [ ] Remove backward-compat aliases in quant_toolkit (FactorModel, FactorRun, etc.) once no code references them

## Deployment

- [ ] Add `gunicorn` to `requirements.txt`
- [ ] Create `Dockerfile` for containerized deployment
- [ ] Create `render.yaml` or `Procfile` for PaaS deployment
- [ ] Set up production config (secret key, debug=False)
- [ ] Purchase and configure custom domain
- [ ] Deploy to hosting platform (Render recommended)
- [ ] Verify `quant-toolkit` git dependency installs in production build

## Future Projects

### Dynamic Asset Allocation (Dynamic Programming)

- [ ] Build a dynamic asset allocation problem solver using dynamic programming
- [ ] New blueprint and lib module

### Black-Scholes Upgrade: Monte Carlo + Volatility Surface

- [ ] Add simulation functionality to the existing Black-Scholes calculator
- [ ] Incorporate Monte Carlo simulation using the underlying risk model analysis
- [ ] Compare the volatility surface of price differences (goal: flatter vol surface)
- [ ] Depends on: Risk Model module

### Machine Learning Forecasting

- [ ] Build ML-based forecasting for asset class returns
- [ ] Built on top of the risk model and regime detection analysis
- [ ] Depends on: Risk Model, regime detection

### Optimization & Backtesting Library

- [ ] Portfolio optimization library with backtesting functionality for dynamic asset allocation
- [ ] Depends on: ML Forecasting, Benchmark Style Analysis, Risk Model
