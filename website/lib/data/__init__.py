"""Data access helpers."""

from .french_industry import (
    SUPPORTED_FACTOR_SETS,
    SUPPORTED_INDUSTRY_UNIVERSES,
    FactorSet,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)

__all__ = [
    "SUPPORTED_FACTOR_SETS",
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "FactorSet",
    "fetch_ff_factors_daily",
    "fetch_ff_industry_daily",
]
