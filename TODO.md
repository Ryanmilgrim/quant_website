# TODO — Quant Dashboard Roadmap

## Priority 1: Deployment (Current Focus)

- [ ] Add `gunicorn` to `requirements.txt`
- [ ] Create `Dockerfile` for containerized deployment
- [ ] Create `render.yaml` or `Procfile` for PaaS deployment
- [ ] Set up production config (secret key, debug=False)
- [ ] Purchase and configure custom domain
- [ ] Deploy to hosting platform (Render recommended)
- [ ] Verify `quant-toolkit` git dependency installs in production build

## Priority 2: BNY-Inspired Theme Refresh

- [ ] Define CSS custom properties for the full BNY-inspired color palette
- [ ] Update `styles.css` to use new color tokens throughout
- [ ] Refine nav bar to match BNY's clean corporate-modern aesthetic
- [ ] Update card and panel styles (teal accents, warm grays)
- [ ] Review and refine typography hierarchy (Oswald headings, Source Sans body)
- [ ] Ensure mobile responsiveness after theme changes
- [ ] Visit bny.com/wealth CMA page for direct visual reference during implementation

## Priority 3: New Analysis Modules

- [ ] Monte Carlo simulation tool
- [ ] Value-at-Risk (VaR) calculator
- [ ] Mean-variance portfolio optimization
- [ ] Additional backtesting frameworks
- [ ] Explore adding more data sources beyond Fama-French

## Priority 4: Content & Polish

- [ ] Write better descriptions for each module on the homepage
- [ ] Add methodology documentation pages for each analysis tool
- [ ] Improve chart interactivity and default Plotly layouts
- [ ] Add loading states for long-running analyses (factor model, style tracker)
- [ ] SEO basics (meta descriptions, Open Graph tags)

## Someday / Maybe

- [ ] Live market data integration (Yahoo Finance daily feeds)
- [ ] PDF export for analysis results
- [ ] Dark mode toggle
- [ ] Performance profiling (large universe + many factors can be slow)
