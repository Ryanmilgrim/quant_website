# agents.md — Agent Instructions for Quant Dashboard

## Project Context

This is Ryan Milgrim's personal quant finance website. Read `CLAUDE.md` for full project context, architecture, and conventions before making any changes.

## Agent Roles

### Frontend / Design Agent
- Reference the BNY-inspired color palette and typography defined in `CLAUDE.md`
- All styling goes in `website/web/static/styles.css` — single file, no preprocessor
- Use CSS custom properties for colors (see Design System in CLAUDE.md)
- Bootstrap 5 grid for layout; custom CSS for component styling
- Plotly.js for all charting — do not introduce other chart libraries
- Templates extend `base.html`; follow the section-numbering pattern
- Keep the editorial research-report aesthetic: numbered sections, kickers, generous whitespace

### Backend / Analysis Agent
- New analysis modules go in `website/lib/analysis/` — pure Python, no Flask imports
- New data sources go in `website/lib/data/`
- New route handlers get their own blueprint in `website/web/blueprints/<feature>/`
- Register new blueprints in `website/web/__init__.py` (`create_app()`)
- Use `lru_cache` for expensive computations and data fetches
- The companion library `quant_toolkit` is available — check its API before reimplementing anything

### Deployment Agent
- No Dockerfile or Procfile exists yet — these need to be created
- The app runs via `python wsgi.py` (calls `create_app()`)
- Production server: use `gunicorn` with the `wsgi:app` entry point
- Environment variable: `QDASH_SECRET_KEY` must be set in production
- The `quant-toolkit` dependency is installed from a git URL — ensure the build environment can access GitHub
- No database to configure — the app is stateless (in-memory cache + file-based result storage)
- `instance/` directory is git-ignored and created at runtime

## Key Files to Know

| File | Purpose |
|------|---------|
| `wsgi.py` | WSGI entry point |
| `website/web/__init__.py` | Flask app factory |
| `website/web/static/styles.css` | All custom CSS |
| `website/web/templates/base.html` | Master template (nav, footer, CDN imports) |
| `website/lib/analysis/` | Pure Python analysis modules |
| `website/lib/data/french_industry.py` | Ken French data fetcher |
| `requirements.txt` | Python dependencies |

## Guard Rails

1. **Never add Flask imports to `website/lib/`** — this layer must remain framework-agnostic
2. **Never remove the disclaimer banner** — it states this is not client advice
3. **Never introduce a JS build step** — all JS is inline or via CDN
4. **Never add a database** unless explicitly requested
5. **Never add user authentication** unless explicitly requested
6. **Keep the BNY-inspired aesthetic** — teal accents, navy dark surfaces, editorial layout
7. **Preserve the `quant_toolkit` integration** — it is the companion library, not a vendored dependency
