from datetime import timedelta

import pandas as pd
import pytest

from quant_research_lab.analytics import features
from quant_research_lab.data import MarketDataset, market_time, synthetic_dataset


def test_indexed_scores_equal_reference_with_delays_and_stale_symbols():
    original = synthetic_dataset(sessions=100)
    frame = original.frame.copy()
    frame.loc[frame.ts_code == "SYN_A", "available_at"] += pd.Timedelta(days=4)
    frame = frame[
        ~(
            (frame.ts_code == "SYN_B")
            & (frame.trade_date > frame.trade_date.max() - pd.Timedelta(days=9))
        )
    ]
    data = MarketDataset(frame, original.manifest)
    for day in data.dates[30::7]:
        cutoff = market_time(day)
        for start in [None, day - timedelta(days=20)]:
            panel = data.view(cutoff).panel()
            if start:
                panel = panel[panel.trade_date >= pd.Timestamp(start)]
            for window in [2, 5, 20]:
                assert data.scores(cutoff, window, start) == features(panel, window)
    with pytest.raises(ValueError):
        data.scores(cutoff.replace(tzinfo=None), 2)


def test_indexed_bars_return_independent_copy():
    data = synthetic_dataset(sessions=100)
    day = data.dates[0]
    expected = data.bars(day)
    changed = data.bars(day)
    changed.loc[:, "close"] = 1.0
    pd.testing.assert_frame_equal(data.bars(day), expected)


def test_historical_eligibility_is_applied_at_each_decision(dataset, config):
    from quant_research_lab.analytics import features
    from quant_research_lab.research import baseline_weights, run_research

    frame = dataset.frame.copy()
    frame["universe_member"] = False
    transition = dataset.dates[40]
    frame.loc[
        (frame.ts_code == "SYN_A") & (frame.trade_date >= pd.Timestamp(transition)),
        "universe_member",
    ] = True
    restricted = MarketDataset(frame, dataset.manifest)
    assert restricted.view(market_time(dataset.dates[39])).universe() == ()
    assert restricted.view(market_time(transition)).universe() == ("SYN_A",)
    for day in [dataset.dates[39], transition, dataset.dates[50]]:
        view = restricted.view(market_time(day))
        assert restricted.scores(market_time(day), 5) == features(view.panel(), 5)
    result = run_research(restricted, config)
    assert all(set(row["approved_weights"]) <= {"SYN_A"} for row in result["decision_log"])
    assert baseline_weights((), config.policy) == {}


def test_next_session_visibility_requires_explicit_signal_age_allowance(dataset, config):
    from quant_research_lab.agents import Committee
    from quant_research_lab.risk import RiskState

    frame = dataset.frame.copy()
    sequence = {day: index + 100.0 for index, day in enumerate(dataset.dates)}
    frame["close"] = frame.trade_date.dt.date.map(sequence)
    for name in ["open", "high", "low"]:
        frame[name] = frame.close
    next_days = dict(zip(dataset.dates[:-1], dataset.dates[1:], strict=True))
    next_days[dataset.dates[-1]] = (pd.Timestamp(dataset.dates[-1]) + pd.offsets.BDay()).date()
    frame["available_at"] = frame.trade_date.dt.date.map(lambda day: market_time(next_days[day], 9))
    delayed = MarketDataset(frame, dataset.manifest)
    view = delayed.view(market_time(delayed.dates[50]))
    assert view.signal_age("SYN_A") == 1
    strict = Committee(config).decide(view, 5, RiskState(delayed.symbols), 1)
    assert not strict["approved_weights"]
    permissive = Committee(config.model_copy(update={"max_signal_age_sessions": 1})).decide(
        view, 5, RiskState(delayed.symbols), 1
    )
    assert permissive["approved_weights"]
