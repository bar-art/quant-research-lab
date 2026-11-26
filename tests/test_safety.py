from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_research_lab.agents import Committee, validate_citations
from quant_research_lab.data import AsOfView, FutureDataError, MarketDataset, market_time
from quant_research_lab.models import Evidence, Forecast, Proposal, ResearchConfig, RiskPolicy
from quant_research_lab.risk import RiskEngine, RiskState


def test_future_access_and_delayed_publication(dataset):
    day = dataset.dates[20]
    view = dataset.view(market_time(day))
    assert view.history("SYN_A").trade_date.max().date() == day
    with pytest.raises(FutureDataError):
        view.history("SYN_A", dataset.dates[21])
    changed = dataset.frame.copy()
    mask = (changed.ts_code == "SYN_A") & (changed.trade_date == pd.Timestamp(day))
    changed.loc[mask, "available_at"] = market_time(dataset.dates[25])
    delayed = MarketDataset(changed, dataset.manifest)
    assert delayed.view(market_time(day)).history("SYN_A").trade_date.max().date() < day
    assert (
        delayed.view(market_time(dataset.dates[25])).history("SYN_A", day).trade_date.max().date()
        == day
    )
    with pytest.raises(ValueError):
        AsOfView(dataset.frame, market_time(day).replace(tzinfo=None))


def test_tomorrow_return_cheat_is_blocked(dataset):
    current = dataset.dates[20]
    view = dataset.view(market_time(current))

    def cheating_factor():
        tomorrow = view.history("SYN_A", dataset.dates[21])
        return tomorrow.close.iloc[-1] / tomorrow.close.iloc[-2] - 1

    with pytest.raises(FutureDataError):
        cheating_factor()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "nan",
        "zero",
        "bad_ohlc",
        "naive_time",
        "early_time",
        "unknown_status",
        "bad_limit",
    ],
)
def test_bad_inputs_fail_closed(dataset, mutation):
    frame = dataset.frame.copy()
    if mutation == "missing":
        frame = frame.drop(columns="available_at")
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "nan":
        frame.loc[0, "close"] = np.nan
    elif mutation == "zero":
        frame.loc[0, "open"] = 0
    elif mutation == "bad_ohlc":
        frame.loc[0, "high"] = 1
    elif mutation == "naive_time":
        frame["available_at"] = "2023-01-01 15:00:00"
    elif mutation == "early_time":
        frame["available_at"] = pd.Timestamp("2000-01-01", tz="UTC")
    elif mutation == "unknown_status":
        frame["is_suspended"] = None
    else:
        frame.loc[0, "up_limit"] = -1
    with pytest.raises(ValueError):
        MarketDataset(frame, dataset.manifest)


def proposal(weights):
    return Proposal(weights=weights, evidence_ids=("e1",), summary="test")


@pytest.mark.parametrize(
    "weights,expected",
    [
        ({"SYN_A": 0.7}, "position_cap"),
        ({"OTHER": 0.1}, "unknown_symbol"),
        ({"SYN_A": -0.1}, "invalid_weight"),
        ({"SYN_A": 0.3, "SYN_B": 0.3, "SYN_C": 0.3}, "gross_cap"),
    ],
)
def test_hard_risk_veto(weights, expected):
    approved, violations = RiskEngine(RiskPolicy()).review(
        proposal(weights), RiskState(("SYN_A", "SYN_B", "SYN_C"))
    )
    assert approved == {}
    assert any(expected in text for text in violations)


@pytest.mark.parametrize(
    "state,expected",
    [
        (RiskState(("SYN_A",), drawdown=0.2), "drawdown_circuit"),
        (RiskState(("SYN_A",), daily_return=-0.05), "daily_loss_circuit"),
        (RiskState(("SYN_A",), drawdown=np.nan), "nonfinite_state"),
    ],
)
def test_risk_state(state, expected):
    assert expected in RiskEngine(RiskPolicy()).review(proposal({"SYN_A": 0.2}), state)[1]


def test_schema_rejects_model_override_and_nonfinite():
    with pytest.raises(ValueError):
        Proposal(weights={"SYN_A": np.nan}, evidence_ids=("e",), summary="x")
    with pytest.raises(ValueError):
        Proposal(weights={}, evidence_ids=("e",), summary="x", disable_risk=True)
    with pytest.raises(ValueError):
        ResearchConfig(windows=(5, 5))
    with pytest.raises(ValueError):
        RiskPolicy(max_gross=0.9, min_cash=0.4)


def test_citations_and_clock(dataset):
    stamp = market_time(dataset.dates[20])
    evidence = (Evidence(id="known", available_at=stamp, summary="x"),)
    with pytest.raises(ValueError):
        validate_citations(("invented",), evidence)
    with pytest.raises(ValueError):
        Evidence(id="x", available_at=stamp.replace(tzinfo=None), summary="x")
    with pytest.raises(ValueError):
        Forecast(
            symbol="x", created_at=stamp, due_at=stamp - timedelta(days=1), predicted_return=0.1
        )


def test_stale_market_proposal_blocked(dataset, config):
    view = dataset.view(market_time(dataset.dates[-1]) + timedelta(days=1))
    decision = Committee(config).decide(view, 3, RiskState(dataset.symbols), 3)
    assert decision["approved_weights"] == {}
    if decision["manager"]["weights"]:
        assert any("stale_market_data" in text for text in decision["risk_violations"])
