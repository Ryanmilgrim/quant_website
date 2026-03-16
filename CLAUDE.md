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
      factor/            # Factor Risk Model (/factor)
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

| Role             | Token                | Hex       | Usage                                      |
|------------------|----------------------|-----------|---------------------------------------------|
| Primary teal     | `--color-primary`    | `#00857C` | Buttons, links, accent arrow, active states |
| Primary dark     | `--color-primary-dk` | `#006B64` | Hover states, active nav                    |
| Navy             | `--color-navy`       | `#1C2B3A` | Nav bar, dark panels, footer                |
| Navy deep        | `--color-navy-deep`  | `#0F1924` | Body background, hero sections              |
| Charcoal         | `--color-charcoal`   | `#2A2A2A` | Primary text on light backgrounds           |
| Warm gray        | `--color-gray`       | `#A7A5A6` | Secondary text, borders, muted elements     |
| Light gray       | `--color-gray-lt`    | `#F0EFED` | Card backgrounds, section separators        |
| Off-white        | `--color-off-white`  | `#FAF9F7` | Page background on light sections           |
| White            | `--color-white`      | `#FFFFFF` | Card surfaces, input backgrounds            |
| Gold accent      | `--color-gold`       | `#B07E25` | Subtle accents, kickers, tags               |

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
- Section numbering pattern: `<span class="section-number">01</span>`
- Card-based content modules

### CSS
- Single `styles.css` file (no preprocessor)
- BEM-like class naming: `.app-nav`, `.section-header`, `.report-card`
- CSS custom properties for the color palette
- Mobile-responsive (Bootstrap grid handles most of it)

### JavaScript
- Inline `<script>` blocks in templates (no build step)
- Plotly.js for all charts
- AJAX calls via `fetch()` for lazy-loaded data (factor analysis)

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
