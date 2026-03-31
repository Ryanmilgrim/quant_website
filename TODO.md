# TODO — Quant Dashboard Roadmap

## Recently Completed

- [x] **MarkovAutoregression** — AR Order and Switching AR support in regime signals (Hamilton 1989)
- [x] **Regime signal polish** — Annualized stats, NaN-safe JSON, regime overlay legend, param name remapping after reorder, AIC/BIC near Fit button, frequency inference fallback, "regime signal" UI rename, AR controls layout

## Bugs / Known Issues

- [ ] **Regression summary mismatch on view load** — When viewing a saved signal from the data lake, the regression summary (significant variables table) does not render identically to how it appears on the builder page after fitting. Investigate why the view-mode path produces different output than the fit-preview path.

## Rename: Factor Analysis → Risk Analysis

- [ ] Rename blueprint folder `factor/` → `risk/` (or update route prefix)
- [ ] Rename all references in templates, nav links, and homepage cards
- [ ] Update blueprint registration in `create_app()`
- [ ] Rename `website/lib/analysis/` modules as needed
- [ ] Update URL routes (`/factor` → `/risk` or similar)

## Theme & Styling Cleanup

- [ ] Define CSS custom properties for the BNY-inspired color palette
- [ ] Migrate hardcoded colors in `styles.css` to use the new tokens
- [ ] Ensure mobile responsiveness is solid throughout
- [ ] Minor polish: consistent spacing, card styles, typography hierarchy

## Descriptions & Transparency

- [ ] Add explanation bubbles / info tooltips across all modules for users unfamiliar with the quant_toolkit library
- [ ] Write better descriptions for each module on the homepage
- [ ] Add methodology documentation or inline explanations for each analysis tool
- [ ] Improve chart labeling and interactivity (Plotly defaults)

## Future Projects

### Dynamic Asset Allocation (Dynamic Programming)

- [ ] Build a dynamic asset allocation problem solver using dynamic programming
- [ ] New blueprint and lib module

### Black-Scholes Upgrade: Monte Carlo + Volatility Surface

- [ ] Add simulation functionality to the existing Black-Scholes calculator
- [ ] Incorporate Monte Carlo simulation using the underlying risk model analysis
- [ ] Compare the volatility surface of price differences (goal: flatter vol surface)
- [ ] Depends on: Risk Analysis module

### Machine Learning Forecasting

- [ ] Build ML-based forecasting for asset class returns
- [ ] Built on top of the risk/factor analysis and regime detection analysis
- [ ] Depends on: Risk Analysis, regime detection

### Optimization & Backtesting Library

- [ ] Portfolio optimization library with backtesting functionality for dynamic asset allocation
- [ ] Depends on: ML Forecasting, Benchmark Style Analysis, Risk Analysis

## Deployment

- [ ] Add `gunicorn` to `requirements.txt`
- [ ] Create `Dockerfile` for containerized deployment
- [ ] Create `render.yaml` or `Procfile` for PaaS deployment
- [ ] Set up production config (secret key, debug=False)
- [ ] Purchase and configure custom domain
- [ ] Deploy to hosting platform (Render recommended)
- [ ] Verify `quant-toolkit` git dependency installs in production build
