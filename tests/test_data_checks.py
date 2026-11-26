import pandas as pd
import pytest

from quant_research_lab.data_checks import compare_raw_prices


def prices():
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 2,
            "trade_date": ["2024-01-02", "2024-01-03"],
            **{name: [10.0, 11.0] for name in ["open", "high", "low", "close"]},
        }
    )


def test_price_agreement_and_missing_days():
    frame = prices()
    assert compare_raw_prices(frame, frame)["status"] == "matched"
    assert compare_raw_prices(frame, frame.iloc[:1])["missing_right"] == 1
    changed = frame.copy()
    changed.loc[1, "close"] = 11.02
    assert compare_raw_prices(frame, changed)["price_mismatch_rows"] == 1
    with pytest.raises(ValueError):
        compare_raw_prices(pd.concat([frame, frame]), frame)
