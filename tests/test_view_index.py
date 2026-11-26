from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_research_lab.data import (
    AsOfView,
    FutureDataError,
    MarketDataset,
    market_time,
    synthetic_dataset,
)


def irregular_dataset() -> MarketDataset:
    """发布延迟不单调、某标的中途停更、股票池资格中途变化。"""
    original = synthetic_dataset(sessions=120)
    frame = original.frame.copy()
    rng = np.random.default_rng(7)
    a_rows = frame.ts_code == "SYN_A"
    delays = rng.integers(0, 4, a_rows.sum())
    frame.loc[a_rows, "available_at"] += pd.to_timedelta(delays, unit="D")
    frame = frame[
        ~(
            (frame.ts_code == "SYN_B")
            & (frame.trade_date > frame.trade_date.iloc[-1] - pd.Timedelta(days=20))
        )
    ]
    frame.loc[(frame.ts_code == "SYN_C") & (frame.trade_date.dt.month == 3), "universe_member"] = (
        False
    )
    return MarketDataset(frame.reset_index(drop=True), original.manifest)


def test_indexed_view_matches_reference_implementation():
    data = irregular_dataset()
    for day in data.dates[5::9]:
        for cutoff in [market_time(day), market_time(day) + timedelta(hours=20)]:
            fast = data.view(cutoff)
            reference = AsOfView(data.frame, cutoff, data._session_index)
            assert fast.universe() == reference.universe()
            assert fast.evidence() == reference.evidence()
            for window in [2, 5, 20]:
                assert fast.momentum(window) == reference.momentum(window)
            for symbol in [*data.symbols, "MISSING"]:
                pd.testing.assert_frame_equal(fast.history(symbol), reference.history(symbol))
                assert fast.signal_age(symbol) == reference.signal_age(symbol)
            earlier = data.dates[0]
            pd.testing.assert_frame_equal(
                fast.history("SYN_A", earlier), reference.history("SYN_A", earlier)
            )
            pd.testing.assert_frame_equal(fast.panel(), reference.panel())


def test_indexed_view_keeps_isolation_and_returns_copies():
    data = irregular_dataset()
    day = data.dates[40]
    view = data.view(market_time(day))
    with pytest.raises(FutureDataError):
        view.history("SYN_A", data.dates[41])
    changed = view.history("SYN_D")
    changed.loc[:, "close"] = -1.0
    assert (data.view(market_time(day)).history("SYN_D").close > 0).all()
    assert view.history("SYN_D").trade_date.max().date() <= day
    with pytest.raises(ValueError):
        data.visible_rows(market_time(day).replace(tzinfo=None))


def test_recent_and_count_memories_match_full_query(tmp_path):
    from quant_research_lab.store import AuditStore

    store = AuditStore(tmp_path / "audit.sqlite")
    days = synthetic_dataset(sessions=100).dates[:6]
    for number, day in enumerate(days):
        memories = [{"id": f"m{number}-{k}", "value": k} for k in range(3)]
        store.record_session("run", f"s{number}", market_time(day), {"n": number}, memories)
    for day in days:
        cutoff = market_time(day)
        full = store.memories_as_of("run", cutoff)
        assert store.count_memories_as_of("run", cutoff) == len(full)
        for limit in [1, 4, 20]:
            assert store.recent_memories_as_of("run", cutoff, limit) == full[-limit:]
