# quant_website

Flask web dashboard for interactive quantitative finance analysis with Plotly.js.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python wsgi.py
```

## Project structure

```
website/
  lib/                   # Pure library code (analysis, data, universe)
  web/                   # Flask application
    blueprints/          # Route blueprints (core, options, style, universe)
    services/            # Background services (universe cache)
    static/              # CSS and static assets
    templates/           # Jinja2 HTML templates
wsgi.py                  # WSGI entrypoint
requirements.txt         # Python dependencies
```

## Companion library

See [quant_toolkit](https://github.com/Ryanmilgrim/quant_toolkit) for the pure-Python quantitative finance toolkit.
