"""WSGI entrypoint for the quant website Flask app."""

from website.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
