"""收益口径：包含起始净值，零波动指标返回空值。"""

import math
from typing import Any

import numpy as np
import pandas as pd


def performance(values: list[float], annual_sessions: int = 252) -> dict[str, float | None]:
    nav = np.asarray(values, dtype=float)
    if len(nav) < 2 or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("绩效输入必须包含至少两个正的有限净值")
    returns = nav[1:] / nav[:-1] - 1
    volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(annual_sessions)) if len(returns) > 1 else 0.0
    )
    total = float(nav[-1] / nav[0] - 1)
    annual = float((nav[-1] / nav[0]) ** (annual_sessions / len(returns)) - 1)
    max_dd = float(np.max(1 - nav / np.maximum.accumulate(nav)))
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(annual_sessions))
        if volatility > 1e-12
        else None
    )
    return {
        "total_return": total,
        "annual_return": annual,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": annual / max_dd if max_dd > 1e-12 else None,
    }


def features(frame: pd.DataFrame, window: int) -> dict[str, float]:
    result = {}
    for symbol, group in frame.groupby("ts_code"):
        ordered = group.sort_values("trade_date")
        if len(ordered) <= window:
            continue
        last = ordered.iloc[-1]
        if "universe_member" in ordered and not last.universe_member:
            continue
        if last.trade_date != frame.trade_date.max() or last.volume <= 0 or last.is_suspended:
            continue
        result[str(symbol)] = float(last.close / ordered.iloc[-window - 1].close - 1)
    return result


def target_weights(
    scores: dict[str, float], top_k: int, cap: float, gross: float
) -> dict[str, float]:
    selected = [
        symbol for symbol in sorted(scores, key=lambda s: (-scores[s], s)) if scores[symbol] > 0
    ][:top_k]
    if not selected:
        return {}
    weight = min(cap, gross / len(selected))
    return dict.fromkeys(selected, weight)


def bootstrap_scenarios(
    returns: list[float], seed: int, paths: int, horizon: int
) -> dict[str, Any]:
    sample = np.asarray(returns, dtype=float)
    if len(sample) < 10 or not np.isfinite(sample).all() or (sample <= -1).any():
        raise ValueError("情景模拟需要至少10个有效训练收益")
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(paths, horizon), replace=True)
    cumulative = np.prod(1 + draws, axis=1) - 1
    return {
        "method": "iid_training_return_bootstrap_not_forecast",
        "seed": seed,
        "paths": paths,
        "horizon_sessions": horizon,
        "p05": float(np.quantile(cumulative, 0.05)),
        "median": float(np.median(cumulative)),
        "p95": float(np.quantile(cumulative, 0.95)),
        "limitation": "独立重采样不保留波动聚集或极端事件结构；不是未来分布保证",
    }


def probabilistic_sharpe(returns: list[float], threshold: float = 0.0) -> float | None:
    """Bailey & López de Prado (2012) PSR：在样本长度、偏度和峰度下，真实单期夏普高于阈值的概率。"""
    sample = np.asarray(returns, dtype=float)
    if len(sample) < 3 or not np.isfinite(sample).all():
        return None
    std = float(np.std(sample, ddof=1))
    if std <= 1e-12:
        return None
    sharpe = float(np.mean(sample)) / std
    centered = sample - np.mean(sample)
    population_std = float(np.sqrt(np.mean(centered**2)))
    skew = float(np.mean(centered**3)) / population_std**3
    kurtosis = float(np.mean(centered**4)) / population_std**4
    variance = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if variance <= 0:
        return None
    z = (sharpe - threshold) * np.sqrt(len(sample) - 1) / np.sqrt(variance)
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))


def relative_performance(
    values: list[float], baseline: list[float], annual_sessions: int = 252
) -> dict[str, float | None]:
    """策略相对同口径基线：超额累计收益、主动收益的年化跟踪误差与信息比率。"""
    nav, base = np.asarray(values, dtype=float), np.asarray(baseline, dtype=float)
    if len(nav) != len(base) or len(nav) < 2:
        raise ValueError("策略与基线净值长度必须一致且至少两个点")
    active = (nav[1:] / nav[:-1]) - (base[1:] / base[:-1])
    tracking = float(np.std(active, ddof=1) * np.sqrt(annual_sessions)) if len(active) > 1 else 0.0
    return {
        "excess_total_return": float(nav[-1] / nav[0] - base[-1] / base[0]),
        "tracking_error": tracking,
        "information_ratio": (
            float(np.mean(active) * annual_sessions / tracking) if tracking > 1e-12 else None
        ),
        "active_sessions_positive": float(np.mean(active > 0)),
    }


def trading_summary(
    trades: list[dict[str, Any]], values: list[float], annual_sessions: int = 252
) -> dict[str, float]:
    """成交额相对平均权益的换手，以及佣金、滑点、税费合计占起始资金的比例。"""
    nav = np.asarray(values, dtype=float)
    traded = sum(item["quantity"] * item["price"] for item in trades)
    costs = sum(
        item["commission"] + item["slippage_cost"] + item.get("stamp_duty", 0.0) for item in trades
    )
    periods = max(len(nav) - 1, 1)
    # 单边换手：买卖成交额之和的一半除以平均权益。
    turnover = traded / 2 / float(np.mean(nav))
    return {
        "trade_count": float(len(trades)),
        "one_way_turnover": turnover,
        "annual_one_way_turnover": turnover * annual_sessions / periods,
        "total_cost_fraction": costs / float(nav[0]),
    }


def forecast_summary(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """到期预测：方向命中率、平均绝对误差，以及按创建日计算的横截面秩相关（Rank IC）均值。"""
    if not scores:
        return {"count": 0.0, "hit_rate": None, "mae": None, "rank_ic_mean": None, "ic_dates": 0.0}
    frame = pd.DataFrame(scores)
    ics = []
    for _, group in frame.groupby("created_at"):
        if len(group) < 3:
            continue
        predicted = group.predicted_return.rank()
        realized = group.realized_return.rank()
        if predicted.nunique() < 2 or realized.nunique() < 2:
            continue
        ics.append(float(np.corrcoef(predicted, realized)[0, 1]))
    return {
        "count": float(len(frame)),
        "hit_rate": float(frame.direction_correct.mean()),
        "mae": float(frame.absolute_error.mean()),
        "rank_ic_mean": float(np.mean(ics)) if ics else None,
        "ic_dates": float(len(ics)),
    }
