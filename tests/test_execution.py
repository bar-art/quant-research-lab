from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_research_lab.data import market_time
from quant_research_lab.execution import PaperBook


def bar(**updates):
    values = {
        "ts_code": "A",
        "open": 10.0,
        "close": 10.0,
        "is_suspended": False,
        "price_limit_exempt": True,
        "up_limit": np.nan,
        "down_limit": np.nan,
    }
    values.update(updates)
    return pd.DataFrame([values])


def test_hand_calculated_fees_and_cash():
    book = PaperBook(commission_bps=10, slippage_bps=0, initial_cash=1000)
    book.rebalance({"A": 0.5}, bar(), date(2024, 1, 3), market_time(date(2024, 1, 2)))
    assert book.quantity("A") == 50
    assert book.cash == pytest.approx(499.5)
    assert book.mark_close(bar(close=11)) == pytest.approx(1049.5)
    book.rebalance({}, bar(open=12, close=12), date(2024, 1, 4), market_time(date(2024, 1, 3)))
    assert book.cash == pytest.approx(1098.9)
    assert sum(item["commission"] for item in book.trades) == pytest.approx(1.1)


def test_same_day_signal_and_sell_rejected():
    book = PaperBook(0, 0, 1000)
    day, previous = date(2024, 1, 3), market_time(date(2024, 1, 2))
    with pytest.raises(ValueError):
        book.rebalance({"A": 0.5}, bar(), day, market_time(day))
    book.rebalance({"A": 0.5}, bar(), day, previous)
    book.rebalance({}, bar(), day, previous)
    assert book.quantity("A") == 50
    assert book.blocks[-1]["reason"] == "t_plus_one"


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"is_suspended": True}, "suspended"),
        ({"price_limit_exempt": False}, "unknown_price_limits"),
        ({"price_limit_exempt": False, "up_limit": 10.0, "down_limit": 9.0}, "up_limit_at_open"),
    ],
)
def test_no_fake_buys(updates, reason):
    book = PaperBook(0, 0, 1000)
    book.rebalance({"A": 0.5}, bar(**updates), date(2024, 1, 3), market_time(date(2024, 1, 2)))
    assert book.trades == []
    assert book.blocks[-1]["reason"] == reason


def test_down_limit_missing_and_suspended_valuation():
    book = PaperBook(0, 0, 1000)
    book.rebalance({"A": 0.5}, bar(), date(2024, 1, 3), market_time(date(2024, 1, 2)))
    book.mark_close(bar())
    assert book.value(bar(close=1, is_suspended=True)) == 1000
    book.rebalance(
        {},
        bar(open=9, close=9, down_limit=9.0, up_limit=11.0, price_limit_exempt=False),
        date(2024, 1, 4),
        market_time(date(2024, 1, 3)),
    )
    assert book.blocks[-1]["reason"] == "down_limit_at_open"
    book.rebalance({}, bar().iloc[:0], date(2024, 1, 5), market_time(date(2024, 1, 4)))
    assert book.blocks[-1]["reason"] == "missing_bar"


def test_cash_and_finite_guard():
    book = PaperBook(5, 5, 1000)
    book.rebalance({"A": 1.0}, bar(), date(2024, 1, 3), market_time(date(2024, 1, 2)))
    assert book.cash >= -1e-10
    assert book.trades[-1]["slippage_cost"] > 0
    for targets in [{"A": np.nan}, {"A": -1.0}, {"A": float("inf")}, {"A": 2.0}]:
        with pytest.raises(ValueError):
            book.rebalance(targets, bar(), date(2024, 1, 4), market_time(date(2024, 1, 3)))
