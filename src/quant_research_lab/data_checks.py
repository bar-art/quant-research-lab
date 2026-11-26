"""Two-source raw-price reconciliation; agreement is not original-vintage certification."""

from typing import Any

import numpy as np
import pandas as pd


def compare_raw_prices(
    left: pd.DataFrame, right: pd.DataFrame, *, tolerance: float = 0.011
) -> dict[str, Any]:
    keys, columns = ["ts_code", "trade_date"], ["open", "high", "low", "close"]
    prepared = []
    for frame in [left, right]:
        if set(keys + columns) - set(frame):
            raise ValueError("Raw OHLC price keys missing")
        data = frame[keys + columns].copy()
        data["trade_date"] = pd.to_datetime(data.trade_date).dt.normalize()
        if data[keys].isna().any().any() or data.duplicated(keys).any():
            raise ValueError("Missing or duplicate price keys")
        for column in columns:
            data[column] = pd.to_numeric(data[column], errors="raise")
            if not np.isfinite(data[column]).all() or (data[column] <= 0).any():
                raise ValueError("Invalid raw prices")
        prepared.append(data)
    merged = prepared[0].merge(
        prepared[1],
        on=keys,
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate="one_to_one",
    )
    common = merged[merged["_merge"] == "both"].copy()
    differences = {
        column: (common[column + "_left"] - common[column + "_right"]).abs() for column in columns
    }
    bad = pd.Series(False, index=common.index)
    for delta in differences.values():
        bad |= delta > tolerance
    missing_left, missing_right = (
        int((merged["_merge"] == "right_only").sum()),
        int((merged["_merge"] == "left_only").sum()),
    )
    return {
        "status": "matched"
        if len(common) and not bad.any() and not missing_left and not missing_right
        else "mismatch",
        "matched_keys": len(common),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "price_mismatch_rows": int(bad.sum()),
        "absolute_tolerance": tolerance,
        "max_absolute_difference": {
            name: float(delta.max()) if len(delta) else None for name, delta in differences.items()
        },
        "examples": common.loc[bad, keys].astype(str).head(20).to_dict(orient="records"),
        "note": "Raw inputs required; price agreement does not certify corporate actions",
    }
