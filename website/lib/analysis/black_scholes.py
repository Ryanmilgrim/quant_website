from math import erf, exp, log, sqrt


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for a standard normal variable."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Compute the Black-Scholes price for a European option."""
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("Inputs must be positive")

    d1 = (
        log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))
    d2 = d1 - volatility * sqrt(time_to_expiry)

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2)

    if option_type == "put":
        return strike * exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")
