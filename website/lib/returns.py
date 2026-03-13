from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(simple: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert simple returns to log returns, masking invalid values."""
    out = simple.copy()
    bad = out <= -1.0
    if np.any(bad.to_numpy()):
        out = out.mask(bad, np.nan)
    return np.log1p(out)
