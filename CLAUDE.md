# CLAUDE.md — Quant Dashboard

## Project Overview

**Quant Dashboard** is a Flask-based interactive quantitative finance website created by Ryan Milgrim, CFA. It serves as a personal web presence that showcases quant expertise to hiring managers, industry peers, finance students, and the general public. It is not a commercial product — it is a demonstration of strong understanding of quantitative finance topics, built for education and professional visibility.

**Companion library:** [quant_toolkit](https://github.com/Ryanmilgrim/quant_toolkit) — pure-Python quantitative finance toolkit (installed from git).

## Architecture

```
website/
  lib/                   # Pure library code — NO Flask imports
    analysis/            # Black-Scholes, benchmark style, factor models
    data/                # Data fetching (Ken French FTP)
    universe/            # Universe loader utilities
    returns.py           # Return transformation helpers
  web/                   # Flask application layer
    __init__.py          # App factory: create_app()
    blueprints/          # Route handlers by feature
      core/              # Homepage (/)
      universe/          # Investment Universe (/universe)
      options/           # Black-Scholes pricing (/options)
      style/             # Benchmark Style analysis (/style)
      risk/              # Risk Model (/risk)
      regime/            # Regime Signals (/regime)
    services/            # Background services (universe cache)
    static/              # CSS (styles.css)
    templates/           # Jinja2 HTML templates
wsgi.py                  # WSGI entrypoint — run with: python wsgi.py
requirements.txt         # Python dependencies
```

### Key design decisions
- **lib/ vs web/ separation**: Library code in `lib/` must remain framework-agnostic (no Flask imports). All Flask-specific code lives in `web/`.
- **No database**: Data is fetched from Ken French's FTP server and cached in-memory via `lru_cache`. Analysis results are saved as files in `instance/`.
- **CDN dependencies**: Bootstrap 5.3, Plotly.js, MathJax are loaded from CDNs — no npm/webpack build step.

## Running Locally

```bash
pip install -r requirements.txt
python wsgi.py
```

Environment variables:
- `QDASH_SECRET_KEY` — Flask session secret (defaults to `"dev-secret-key"` in development)

## Design System — BNY-Inspired Theme

The visual design is inspired by BNY's (Bank of New York) 2024 rebrand and their Capital Market Assumptions website. The goal is a polished, corporate-modern aesthetic that feels like a professional research publication.

### Color palette (BNY-inspired, not an exact copy)

| Role             | CSS Variable           | Hex       | Usage                                      |
|------------------|------------------------|-----------|---------------------------------------------|
| Navy deep        | `--ink-900`            | `#00233c` | Primary text, headings, dark accents        |
| Navy             | `--ink-800`            | `#001f38` | Font color, axis labels                     |
| Teal dark        | `--teal-700`           | `#00475e` | Deep teal accents                           |
| Teal hover       | `--teal-600`           | `#45999a` | Button/link hover states                    |
| Teal primary     | `--teal-500`           | `#2c9bac` | Buttons, links, active states, section numbers |
| Teal light       | `--teal-400`           | `#6abcc5` | Charts, secondary accents                   |
| Teal muted       | `--teal-300`           | `#73afb0` | Panel kickers on dark backgrounds           |
| White            | `--paper`              | `#ffffff` | Card surfaces, input backgrounds            |
| Surface          | `--surface`            | `#eaeaea` | Card backgrounds                            |
| Surface alt      | `--surface-2`          | `#f4f4f5` | Form backgrounds, hero cards                |
| Border           | `--border`             | `#d8dadb` | Card borders, dividers, grid lines          |
| Muted            | `--muted`              | `#667d89` | Secondary text, kickers, labels             |
| Support          | `--support`            | `#285064` | Body paragraph text                         |
| Accent           | `--accent`             | `#f86018` | Sparingly, call-to-action highlights        |

**Chart-specific tokens** (defined in `QD_THEME` JS object in `base.html`):
- Forecast/Predicted: `#e63946` (red)
- Realized/Trailing: `#457b9d` (slate blue)
- Confidence bands: `#a8dadc` (light teal)

### Typography

| Role      | Font stack                             | Weight   | Usage                        |
|-----------|----------------------------------------|----------|-------------------------------|
| Headings  | `"Oswald", sans-serif`                 | 500–700  | Section titles, hero headlines |
| Body      | `"Source Sans 3", sans-serif`          | 400–600  | Paragraphs, labels, nav links |
| Serif     | `"Merriweather", serif`               | 400–700  | Pull quotes, editorial accent |

### Design principles
1. **Research-report feel** — Numbered sections, kickers, editorial layout reminiscent of a published CMA report
2. **Dark nav + light content** — Navy navigation bar, off-white/white content area
3. **Teal as the action color** — All interactive elements (buttons, links, active states) use the primary teal
4. **Generous whitespace** — Spacious card layouts, clear visual hierarchy
5. **Data-forward** — Plotly charts are the centerpiece; UI stays out of the way

### Current fonts loaded (Google Fonts)
```
Merriweather:wght@400;700
Oswald:wght@500;600;700
Source+Sans+3:wght@400;500;600
```

## Coding Conventions

### Python
- Flask blueprints for route organization, one blueprint per feature module
- App factory pattern (`create_app()`)
- No ORMs — the project has no database layer
- Use `lru_cache` for expensive data fetches
- Keep `lib/` pure — no Flask, no request context
- Type hints encouraged but not enforced
- Follow PEP 8

### HTML / Templates
- Jinja2 extends `base.html` for all pages
- Bootstrap 5 grid for layout
- Section numbering: 01 (homepage hero), 02 (Investment Universe / homepage modules), 03 (Black-Scholes), 04 (Benchmark Style), 05 (Regime Signals), 06 (Risk Model)
- Card-based content modules
- Each module page has a collapsible "Model Overview" section (`.model-description` class, Bootstrap collapse, MathJax formulas)
- Form labels include `.info-tip` tooltips for user guidance

### CSS
- Single `styles.css` file (no preprocessor)
- BEM-like class naming: `.app-nav`, `.section-header`, `.report-card`
- CSS custom properties for the color palette
- Mobile-responsive (Bootstrap grid handles most of it)

### JavaScript
- Inline `<script>` blocks in templates (no build step)
- Plotly.js for all charts
- AJAX calls via `fetch()` for lazy-loaded data (factor analysis)
- **Plotly theme**: `base.html` defines `QD_THEME` (color constants), `QD_PLOTLY_BASE` (shared layout), and `QD_PLOTLY_CONFIG` (toolbar config). Templates use `structuredClone(QD_PLOTLY_BASE)` and override only chart-specific properties. Do not duplicate the full layout object in individual templates.
- **Tooltips**: CSS-only via `.info-tip` class with `data-tooltip` attribute. No JS required.

## Deployment

**Status:** Not yet deployed. Deployment configuration is a current priority.

**Target:** Render, Railway, or similar PaaS with custom domain support.

**Requirements for deployment:**
- Production WSGI server (gunicorn)
- Dockerfile or platform-specific config
- `QDASH_SECRET_KEY` set as environment variable
- The `quant-toolkit` git dependency must be installable in the build environment

## What Not to Do

- Do not add a database unless explicitly requested
- Do not introduce npm, webpack, or any JS build tooling
- Do not replace CDN-loaded libraries with local bundles
- Do not add user authentication (no login system planned)
- Do not change the lib/web separation — keep library code framework-agnostic
- Do not add emojis to the UI
