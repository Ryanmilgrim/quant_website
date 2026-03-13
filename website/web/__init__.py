"""Flask presentation layer for the quant dashboard."""

import os

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = os.environ.get("QDASH_SECRET_KEY", "dev-secret-key")

    from .blueprints.core.routes import core_bp
    from .blueprints.options.routes import options_bp
    from .blueprints.style.routes import style_bp
    from .blueprints.universe.routes import universe_bp

    app.register_blueprint(core_bp)
    app.register_blueprint(universe_bp)
    app.register_blueprint(options_bp)
    app.register_blueprint(style_bp)

    return app


__all__ = ["create_app"]
