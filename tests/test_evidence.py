import numpy as np
import pytest

from quant_research_lab.analytics import (
    forecast_summary,
    probabilistic_sharpe,
    relative_performance,
    trading_summary,
)
from quant_research_lab.data import market_time, synthetic_dataset
from quant_research_lab.execution import PaperBook
from quant_research_lab.models import ResearchConfig, RiskPolicy
from quant_research_lab.reporting import render_html
from quant_research_lab.research import baseline_weights, run_research


def test_probabilistic_sharpe_hand_cases():
    # 均值为0、对称收益：夏普为0，PSR 恰为0.5。
    assert probabilistic_sharpe([0.01, -0.01] * 20) == pytest.approx(0.5)
    rng = np.random.default_rng(1)
    positive = (rng.normal(0.002, 0.01, 500)).tolist()
    assert probabilistic_sharpe(positive) > 0.9
    assert probabilistic_sharpe([-x for x in positive]) < 0.1
    assert probabilistic_sharpe([0.01] * 10) is None
    assert probabilistic_sharpe([0.01, 0.02]) is None


def test_relative_performance_hand_example():
    stats = relative_performance([100, 110, 121], [100, 105, 105])
    # 累计 21% 对 5%；主动收益 [0.05, 0.10]。
    assert stats["excess_total_return"] == pytest.approx(0.16)
    assert stats["active_sessions_positive"] == 1.0
    assert stats["tracking_error"] == pytest.approx(np.std([0.05, 0.10], ddof=1) * np.sqrt(252))
    with pytest.raises(ValueError):
        relative_performance([100, 101], [100])


def test_trading_summary_hand_example():
    trades = [
        {"quantity": 10, "price": 100, "commission": 0.5, "slippage_cost": 0.2},
        {
            "quantity": 10,
            "price": 110,
            "commission": 0.55,
            "slippage_cost": 0.22,
            "stamp_duty": 0.55,
        },
    ]
    stats = trading_summary(trades, [10000, 10000, 10000])
    assert stats["one_way_turnover"] == pytest.approx(2100 / 2 / 10000)
    assert stats["annual_one_way_turnover"] == pytest.approx(2100 / 2 / 10000 * 252 / 2)
    assert stats["total_cost_fraction"] == pytest.approx(2.02 / 10000)


def test_forecast_summary_rank_ic():
    rows = []
    for day, sign in [("d1", 1), ("d2", -1)]:
        for k, symbol in enumerate("ABC"):
            rows.append(
                {
                    "created_at": day,
                    "symbol": symbol,
                    "predicted_return": 0.01 * k,
                    "realized_return": sign * 0.02 * k,
                    "absolute_error": 0.01,
                    "direction_correct": sign > 0,
                }
            )
    stats = forecast_summary(rows)
    assert stats["rank_ic_mean"] == pytest.approx(0.0)  # +1 与 -1 平均
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["ic_dates"] == 2
    assert forecast_summary([])["hit_rate"] is None


def test_stamp_duty_hand_calculation(dataset):
    day0, day1, day2 = dataset.dates[:3]
    book = PaperBook(0, 0, 100000, stamp_duty_bps=5)
    book.rebalance({"SYN_A": 0.3}, dataset.bars(day1), day1, market_time(day0))
    book.mark_close(dataset.bars(day1))
    quantity = book.quantity("SYN_A")
    cash_before = book.cash
    book.rebalance({}, dataset.bars(day2), day2, market_time(day1))
    price = float(dataset.bars(day2).set_index("ts_code").loc["SYN_A", "open"])
    assert book.cash - cash_before == pytest.approx(quantity * price * (1 - 0.0005))
    assert book.trades[-1]["stamp_duty"] == pytest.approx(quantity * price * 0.0005)
    assert book.trades[0]["stamp_duty"] == 0


def test_no_phantom_block_after_full_sale(dataset):
    d = dataset.dates
    book = PaperBook()
    book.rebalance({"SYN_A": 0.3}, dataset.bars(d[1]), d[1], market_time(d[0]))
    book.mark_close(dataset.bars(d[1]))
    book.rebalance({}, dataset.bars(d[2]), d[2], market_time(d[1]))
    book.mark_close(dataset.bars(d[2]))
    assert book.quantity("SYN_A") == 0 and "SYN_A" not in book.lots
    for k in range(3, 6):
        bars = dataset.bars(d[k])
        bars = bars[bars.ts_code != "SYN_A"]
        book.rebalance({}, bars, d[k], market_time(d[k - 1]))
        book.mark_close(bars)
    assert book.blocks == []


def test_baseline_respects_position_cap():
    policy = RiskPolicy(max_position=0.3, max_gross=0.6, min_cash=0.4)
    assert baseline_weights(("A",), policy) == {"A": 0.3}
    assert baseline_weights(("A", "B", "C", "D"), policy) == dict.fromkeys("ABCD", 0.15)


def test_drawdown_circuit_latch_is_reported_and_reset_is_optional():
    data = synthetic_dataset(42)
    tight = RiskPolicy(max_drawdown=0.04)
    latched = run_research(data, ResearchConfig(policy=tight))
    events = latched["evidence"]["circuit_events"]
    first = next(i for i, s in enumerate(latched["sessions"]) if s["risk_violations"])
    assert events and events[0]["reasons"] == ["drawdown_circuit"]
    # never：首次触发后再无获批持仓，并在局限中明确说明。
    assert not any(s["approved_weights"] for s in latched["sessions"][first:])
    assert any("不再开仓" in text for text in latched["limitations"])
    reset = run_research(data, ResearchConfig(policy=tight, drawdown_reset="next_fold"))
    assert any(s["approved_weights"] for s in reset["sessions"][first + 25 :])
    assert reset["evidence"]["circuit_sessions"] < len(events)


def test_evidence_block_and_report(result):
    evidence = result["evidence"]
    assert evidence["fold_count"] == len(result["folds"])
    assert 0 <= evidence["folds_beating_baseline"] <= evidence["fold_count"]
    for fold in result["folds"]:
        assert fold["beat_baseline"] == (
            fold["oos_metrics"]["total_return"] > fold["baseline_oos_metrics"]["total_return"]
        )
    assert evidence["forecasts"]["count"] == len(result["forecast_scores"])
    assert all("created_at" in item for item in result["forecast_scores"])
    html = render_html(result)
    assert "证据强度" in html and "相对基线超额" in html and "跑赢基线" in html


def test_dated_stamp_duty_schedule(dataset):
    from datetime import date
    from pathlib import Path

    book = PaperBook(
        0, 0, 100000, stamp_duty_schedule=((date(2023, 8, 28), 5), (date(2008, 9, 19), 10))
    )
    assert book.stamp_duty_rate(date(2023, 8, 25)) == 0.001
    assert book.stamp_duty_rate(date(2023, 8, 28)) == 0.0005
    with pytest.raises(ValueError, match="税率未知"):
        book.stamp_duty_rate(date(2008, 9, 18))
    assert PaperBook(stamp_duty_bps=5).stamp_duty_rate(date(2000, 1, 4)) == 0.0005
    config = ResearchConfig.model_validate_json(
        (Path(__file__).parents[1] / "config" / "ashare.json").read_text()
    )
    assert config.stamp_duty_schedule[-1] == (date(2023, 8, 28), 5.0)
    with pytest.raises(ValueError):
        ResearchConfig(stamp_duty_bps=5, stamp_duty_schedule=((date(2023, 8, 28), 5),))
    with pytest.raises(ValueError):
        ResearchConfig(stamp_duty_schedule=((date(2023, 8, 28), 5), (date(2023, 8, 28), 10)))
