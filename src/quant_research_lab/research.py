"""滚动训练选择→样本外纸面回放→到期预测评分。"""

import hashlib
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .agents import Committee, StructuredModel
from .analytics import (
    bootstrap_scenarios,
    forecast_summary,
    performance,
    probabilistic_sharpe,
    relative_performance,
    target_weights,
    trading_summary,
)
from .data import MarketDataset, market_time
from .execution import PaperBook
from .models import Forecast, ResearchConfig, RiskPolicy
from .risk import RiskState
from .store import AuditStore


def new_book(config: ResearchConfig) -> PaperBook:
    return PaperBook(
        config.commission_bps,
        config.slippage_bps,
        stamp_duty_bps=config.stamp_duty_bps,
        stamp_duty_schedule=config.stamp_duty_schedule,
    )


def baseline_weights(symbols: tuple[str, ...], policy: RiskPolicy) -> dict[str, float]:
    """等权基线遵守与策略相同的单票与总仓位上限。"""
    if not symbols:
        return {}
    return dict.fromkeys(symbols, min(policy.max_position, policy.max_gross / len(symbols)))


def simulate_training(
    dataset: MarketDataset, dates: list[date], window: int, config: ResearchConfig
) -> list[float]:
    book = new_book(config)
    values = [book.cash]
    pending: dict[str, float] = {}
    for i, day in enumerate(dates):
        bars = dataset.bars(day)
        if i:
            book.rebalance(pending, bars, day, market_time(dates[i - 1]))
        values.append(book.mark_close(bars))
        # 训练候选的特征只使用训练窗口内数据，不借用窗口之前的观测。
        pending = target_weights(
            dataset.scores(market_time(day), window, start=dates[0]),
            config.top_k,
            config.policy.max_position,
            config.policy.max_gross,
        )
    return values


def select_window(
    dataset: MarketDataset, training_dates: list[date], config: ResearchConfig, fold: int
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    curves: dict[int, list[float]] = {}
    for window in config.windows:
        curve = simulate_training(dataset, training_dates, window, config)
        stats = performance(curve)
        # 训练累计收益减两倍最大回撤；规则事先固定，不用测试集调节。
        score = float(stats["total_return"] or 0) - 2 * float(stats["max_drawdown"] or 0)
        trials.append({"window": window, "score": score, "metrics": stats})
        curves[window] = curve
    selected = int(max(trials, key=lambda trial: (trial["score"], -trial["window"]))["window"])
    curve_array = np.asarray(curves[selected])
    returns = (curve_array[1:] / curve_array[:-1] - 1).tolist()
    scenario = bootstrap_scenarios(
        returns, config.seed + fold, config.scenario_paths, config.scenario_horizon
    )
    return selected, trials, scenario


def run_research(
    dataset: MarketDataset,
    config: ResearchConfig,
    store: AuditStore | None = None,
    *,
    model: StructuredModel | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(dataset.dates) <= config.training_sessions + 1:
        raise ValueError("数据不足以形成训练和样本外区间")
    backend = "llm_committee_replay" if model is not None else "deterministic_rules_not_llm"
    identity = dataset.digest + config.model_dump_json() + backend + str(model_provenance or {})
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    book = new_book(config)
    baseline = new_book(config)
    committee = Committee(config, model)
    decision_log: list[dict[str, Any]] = []
    training_context: dict[str, Any] = {}
    values, baseline_values = [book.cash], [baseline.cash]
    curves: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    pending: dict[str, float] = {}
    previous_date: date | None = None
    selected = config.windows[0]
    all_trials = 0
    peak_from = 0
    circuit_events: list[dict[str, Any]] = []

    def risk_state() -> RiskState:
        # 回撤从 peak_from 起算；drawdown_reset="never" 时始终为0，即全程高点。
        return RiskState(
            dataset.symbols,
            1 - values[-1] / max(values[peak_from:]),
            values[-1] / values[-2] - 1 if len(values) > 1 else 0.0,
        )

    def decide_at(day: date, purpose: str) -> dict[str, Any]:
        context = {
            **training_context,
            "purpose": purpose,
            "portfolio": {
                "cash_fraction": book.cash / values[-1],
                "quantities": {symbol: book.quantity(symbol) for symbol in book.lots},
            },
            "matured_feedback": store.recent_memories_as_of(run_id, market_time(day), 20)
            if store
            else [],
        }
        result = committee.decide(
            dataset.view(market_time(day)), selected, risk_state(), len(config.windows), context
        )
        decision_log.append({"id": len(decision_log), "purpose": purpose, **result})
        if store:
            store.record_decision(run_id, len(decision_log) - 1, market_time(day), decision_log[-1])
        return result

    for index in range(config.training_sessions, len(dataset.dates)):
        day = dataset.dates[index]
        offset = index - config.training_sessions
        if offset % config.test_sessions == 0:
            training = dataset.dates[index - config.training_sessions : index]
            selected, trials, scenarios = select_window(dataset, training, config, len(folds))
            if config.drawdown_reset == "next_fold":
                peak_from = len(values) - 1
            all_trials += len(trials)
            fold: dict[str, Any] = {
                "fold": len(folds),
                "train_start": str(training[0]),
                "train_end": str(training[-1]),
                "test_start": str(day),
                "test_end": str(
                    dataset.dates[min(index + config.test_sessions - 1, len(dataset.dates) - 1)]
                ),
                "selected_window": selected,
                "trials": trials,
                "scenarios": scenarios,
                "start_value_index": len(values) - 1,
                "drawdown_peak_from": peak_from,
            }
            folds.append(fold)
            training_context = {
                "train_start": str(training[0]),
                "train_end": str(training[-1]),
                "candidate_results": trials,
                "scenarios": scenarios,
            }
            # 新窗口先在前一日已知数据上产生提议，最早当日开盘执行。
            previous_date = training[-1]
            pending = decide_at(previous_date, "fold_refresh")["approved_weights"]
        assert previous_date is not None
        bars = dataset.bars(day)
        book.rebalance(pending, bars, day, market_time(previous_date))
        baseline.rebalance(
            baseline_weights(dataset.view(market_time(previous_date)).universe(), config.policy),
            bars,
            day,
            market_time(previous_date),
        )
        values.append(book.mark_close(bars))
        baseline_values.append(baseline.mark_close(bars))
        view = dataset.view(market_time(day))
        decision = decide_at(day, "close")
        pending = decision["approved_weights"]
        circuits = [item for item in decision["risk_violations"] if item.endswith("_circuit")]
        if circuits:
            circuit_events.append({"date": str(day), "reasons": circuits})
        memories = []
        # 只在原定到期日或之后、且后续价格已可得时评分。
        for item in forecasts:
            if item["scored"] or item["due_index"] > index:
                continue
            row = view.history(item["forecast"]["symbol"])
            actual_day = pd.Timestamp(item["forecast"]["due_at"]).tz_convert("Asia/Shanghai").date()
            terminal = row[row.trade_date == pd.Timestamp(actual_day)]
            if terminal.empty:
                continue
            realized = float(terminal.iloc[0].close / item["base_price"] - 1)
            error = realized - item["forecast"]["predicted_return"]
            memory = {
                "id": item["id"],
                "available_at": market_time(day).isoformat(),
                "created_at": item["forecast"]["created_at"],
                "symbol": item["forecast"]["symbol"],
                "predicted_return": item["forecast"]["predicted_return"],
                "realized_return": realized,
                "absolute_error": abs(error),
                "direction_correct": bool(
                    np.sign(realized) == np.sign(item["forecast"]["predicted_return"])
                ),
            }
            scored.append(memory)
            memories.append(memory)
            item["scored"] = True
        # 每个测试段第一日记录一组可核查的5期前瞻预测，不使用情景路径充当真实反馈。
        if offset % config.test_sessions == 0 and index + 5 < len(dataset.dates):
            scores = view.momentum(selected)
            for symbol, score in scores.items():
                forecast = Forecast(
                    symbol=symbol,
                    created_at=market_time(day),
                    due_at=market_time(dataset.dates[index + 5]),
                    predicted_return=score * 5 / selected,
                )
                item = {
                    "id": f"forecast:{day}:{symbol}",
                    "forecast": forecast.model_dump(mode="json"),
                    "base_price": float(view.history(symbol).iloc[-1].close),
                    "due_index": index + 5,
                    "scored": False,
                }
                forecasts.append(item)
                if store:
                    store.save_forecast(run_id, item["id"], forecast.due_at, item)
        if store:
            decision["previous_matured_memories"] = store.count_memories_as_of(
                run_id, market_time(day)
            )
            store.record_session(run_id, str(day), market_time(day), decision, memories)
        sessions.append(decision)
        curves.append(
            {
                "date": str(day),
                "nav": values[-1] / values[0],
                "baseline_nav": baseline_values[-1] / baseline_values[0],
                "cash_fraction": book.cash / values[-1],
                "window": selected,
            }
        )
        previous_date = day
    for i, fold in enumerate(folds):
        end_index = folds[i + 1]["start_value_index"] if i + 1 < len(folds) else len(values) - 1
        window_slice = slice(fold["start_value_index"], end_index + 1)
        fold["oos_metrics"] = performance(values[window_slice])
        fold["baseline_oos_metrics"] = performance(baseline_values[window_slice])
        fold["beat_baseline"] = bool(
            fold["oos_metrics"]["total_return"] > fold["baseline_oos_metrics"]["total_return"]
        )
    nav = np.asarray(values)
    evidence = {
        "relative_to_baseline": relative_performance(values, baseline_values),
        "folds_beating_baseline": sum(fold["beat_baseline"] for fold in folds),
        "fold_count": len(folds),
        "probabilistic_sharpe": probabilistic_sharpe((nav[1:] / nav[:-1] - 1).tolist()),
        "probabilistic_sharpe_note": (
            "单期夏普>0的概率估计，考虑样本长度、偏度与峰度；未扣除跨折参数试验带来的选择偏差。"
        ),
        "trading": trading_summary(book.trades, values),
        "baseline_trading": trading_summary(baseline.trades, baseline_values),
        "forecasts": forecast_summary(scored),
        "circuit_events": circuit_events,
        "circuit_sessions": len(circuit_events),
    }
    limitations = [
        "人工行情只验证工程，不证明盈利。"
        if dataset.manifest.kind == "synthetic"
        else "用户提供数据未由本项目独立核验。",
        "碎股研究撮合：不是完整A股交易规则、税费或公司行动模型。",
        "股票池历史完整性与公司行动状态以manifest声明为准；未作独立审计。",
        "无新闻、财务、行业风险模型；策略是简单正动量规则。",
        (
            "历史回放逐时点使用模型；表现来自本次模型输出，不代表稳定增量。"
            if model
            else "历史回放为规则对照；不能据此推断大模型表现。"
        ),
        "预测反馈记忆只存储/检索，不自动修改策略或风控。",
        "缺失/退市后持仓保留最后已知价并阻断成交，未建模退市清算价值。",
        "停牌、封板会阻止减仓，获批目标不保证实际仓位始终满足上限。",
    ]
    if circuit_events:
        limitations.append(
            f"风控熔断在{len(circuit_events)}个交易日阻止持仓，首次为{circuit_events[0]['date']}；"
            + (
                "drawdown_reset=never，回撤熔断触发后此后不再开仓，后续样本外表现主要是现金。"
                if config.drawdown_reset == "never"
                and any("drawdown_circuit" in e["reasons"] for e in circuit_events)
                else "熔断期间的样本外表现主要是现金。"
            )
        )
    return {
        "run_id": run_id,
        "dataset_sha256": dataset.digest,
        "manifest": dataset.manifest.model_dump(),
        "config": config.model_dump(mode="json"),
        "backtest_backend": backend,
        "model_provenance": model_provenance,
        "decision_log": decision_log,
        "model_meetings": len(decision_log) if model else 0,
        "model_calls_expected": len(decision_log) * 4 if model else 0,
        "forecast_backend": "rule_momentum_diagnostic_not_llm_forecast",
        "parameter_trials": all_trials,
        "multiple_testing_warning": (
            "每折训练内选择参数；多次试验与反复查看样本外结果仍有过拟合风险。"
        ),
        "metrics": performance(values),
        "baseline_metrics": performance(baseline_values),
        "baseline_definition": (
            "按同一成本模型每日调仓、相同总目标仓位与单票上限的全样本标的等权；不是可投资历史指数。"
        ),
        "evidence": evidence,
        "curve": curves,
        "folds": folds,
        "sessions": sessions,
        "trades": book.trades,
        "blocked_orders": book.blocks,
        "forecasts": forecasts,
        "forecast_scores": scored,
        "final_committee": sessions[-1],
        "limitations": limitations,
        "disclaimer": "仅供研究，不构成投资建议",
    }
