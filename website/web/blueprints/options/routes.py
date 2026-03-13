from typing import Optional

from flask import Blueprint, flash, render_template, request

from website.lib.analysis import black_scholes_price

options_bp = Blueprint("options", __name__, url_prefix="/options")


@options_bp.route("/black-scholes", methods=["GET", "POST"])
def black_scholes_view():
    price: Optional[float] = None
    inputs = {
        "underlying_price": request.form.get("underlying_price", ""),
        "strike_price": request.form.get("strike_price", ""),
        "time_to_expiry": request.form.get("time_to_expiry", ""),
        "risk_free_rate": request.form.get("risk_free_rate", ""),
        "volatility": request.form.get("volatility", ""),
        "option_type": request.form.get("option_type", "call"),
    }

    if request.method == "POST":
        try:
            underlying_price = float(inputs["underlying_price"])
            strike_price = float(inputs["strike_price"])
            time_to_expiry = float(inputs["time_to_expiry"])
            risk_free_rate = float(inputs["risk_free_rate"]) / 100
            volatility = float(inputs["volatility"]) / 100
            option_type = inputs["option_type"]

            if option_type not in {"call", "put"}:
                raise ValueError("Invalid option type")

            price = black_scholes_price(
                spot=underlying_price,
                strike=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type,
            )
        except ValueError:
            flash("Please provide valid numeric inputs.", "danger")
        except Exception:
            flash("Unable to calculate option price right now.", "danger")

    return render_template("black_scholes.html", price=price, form_values=inputs)
