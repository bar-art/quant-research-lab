"""研究模拟账本：碎股、下一开盘、先卖后买、购买当日不可卖出。"""

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from .data import market_time


@dataclass(frozen=True)
class ExecutionBar:
    ts_code: str
    open: float
    close: float
    is_suspended: bool
    price_limit_exempt: bool
    up_limit: float
    down_limit: float


EXECUTION_COLUMNS = (
    "ts_code",
    "open",
    "close",
    "is_suspended",
    "price_limit_exempt",
    "up_limit",
    "down_limit",
)


def execution_rows(bars: pd.DataFrame) -> dict[str, ExecutionBar]:
    # 按列取值后逐行组装：逐行 to_dict/itertuples 在每日调用中是主要耗时。
    columns = [bars[name].tolist() for name in EXECUTION_COLUMNS]
    result = {}
    for code, opening, closing, suspended, exempt, up, down in zip(*columns, strict=True):
        bar = ExecutionBar(
            ts_code=str(code),
            open=float(opening),
            close=float(closing),
            is_suspended=bool(suspended),
            price_limit_exempt=bool(exempt),
            up_limit=float(up),
            down_limit=float(down),
        )
        result[bar.ts_code] = bar
    return result


class PaperBook:
    def __init__(
        self,
        commission_bps: float = 5,
        slippage_bps: float = 2,
        initial_cash: float = 100000,
        stamp_duty_bps: float = 0,
        stamp_duty_schedule: tuple[tuple[date, float], ...] = (),
    ) -> None:
        self.cash = float(initial_cash)
        self.commission = commission_bps / 10000
        self.slippage = slippage_bps / 10000
        self.stamp_duty = stamp_duty_bps / 10000
        # (生效日, bps) 按日期升序；设置后按成交日取适用税率，覆盖固定税率。
        self.stamp_duty_schedule = tuple(sorted(stamp_duty_schedule))
        self.lots: dict[str, list[tuple[date, float]]] = {}
        self.last_prices: dict[str, float] = {}
        self.trades: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []

    def stamp_duty_rate(self, day: date) -> float:
        """卖出印花税率；分段表覆盖不到的早期日期直接拒绝，不猜测税率。"""
        if not self.stamp_duty_schedule:
            return self.stamp_duty
        rate = None
        for effective, bps in self.stamp_duty_schedule:
            if effective <= day:
                rate = bps / 10000
        if rate is None:
            raise ValueError(f"{day} 早于印花税分段表的首个生效日，税率未知")
        return rate

    def quantity(self, symbol: str) -> float:
        return sum(quantity for _, quantity in self.lots.get(symbol, []))

    def prices(self, bars: pd.DataFrame, column: str) -> dict[str, float]:
        result = dict(self.last_prices)
        for code, suspended, price in zip(
            bars["ts_code"].tolist(),
            bars["is_suspended"].tolist(),
            bars[column].tolist(),
            strict=True,
        ):
            if not suspended:
                result[str(code)] = float(price)
        return result

    def value(self, bars: pd.DataFrame, column: str = "close") -> float:
        prices = self.prices(bars, column)
        return self.cash + sum(self.quantity(symbol) * prices[symbol] for symbol in self.lots)

    def mark_close(self, bars: pd.DataFrame) -> float:
        value = self.value(bars)
        self.last_prices = self.prices(bars, "close")
        return value

    def rebalance(
        self, targets: dict[str, float], bars: pd.DataFrame, day: date, decided_at: datetime
    ) -> None:
        if decided_at.tzinfo is None or decided_at >= market_time(day, 9):
            raise ValueError("信号必须早于执行日开盘；不得用当日收盘信号成交")
        if (
            any(weight < 0 or not math.isfinite(weight) for weight in targets.values())
            or sum(targets.values()) > 1
        ):
            raise ValueError("账本拒绝负权重、非有限值和杠杆")
        rows = execution_rows(bars)
        equity = self.value(bars, "open")
        desired: dict[str, float] = {}
        for symbol in set(self.lots) | set(targets):
            row = rows.get(symbol)
            if row is None:
                self.blocks.append({"date": str(day), "symbol": symbol, "reason": "missing_bar"})
                continue
            desired[symbol] = targets.get(symbol, 0) * equity / float(row.open)
        # 相同目标和输入始终以同一顺序成交。
        for side in ["sell", "buy"]:
            for symbol in sorted(desired):
                delta = desired[symbol] - self.quantity(symbol)
                if (side == "sell" and delta >= -1e-9) or (side == "buy" and delta <= 1e-9):
                    continue
                row = rows[symbol]
                reason = ""
                if row.is_suspended:
                    reason = "suspended"
                elif not row.price_limit_exempt and (
                    pd.isna(row.up_limit) or pd.isna(row.down_limit)
                ):
                    reason = "unknown_price_limits"
                elif (
                    not row.price_limit_exempt and side == "buy" and row.open >= row.up_limit - 1e-8
                ):
                    reason = "up_limit_at_open"
                elif (
                    not row.price_limit_exempt
                    and side == "sell"
                    and row.open <= row.down_limit + 1e-8
                ):
                    reason = "down_limit_at_open"
                if reason:
                    self.blocks.append({"date": str(day), "symbol": symbol, "reason": reason})
                    continue
                quantity = abs(delta)
                duty = self.stamp_duty_rate(day) if side == "sell" else 0.0
                price = float(row.open) * (1 + self.slippage * (1 if side == "buy" else -1))
                # 含滑点的最终成交价也必须在涨跌停之内：越界即拒单（保守），不裁剪成更有利的价格。
                if not row.price_limit_exempt and (
                    (side == "buy" and price > float(row.up_limit) + 1e-8)
                    or (side == "sell" and price < float(row.down_limit) - 1e-8)
                ):
                    self.blocks.append(
                        {
                            "date": str(day),
                            "symbol": symbol,
                            "reason": "limit_breached_after_slippage",
                        }
                    )
                    continue
                if side == "sell":
                    sellable = sum(q for bought_on, q in self.lots[symbol] if bought_on < day)
                    if quantity > sellable + 1e-9:
                        self.blocks.append(
                            {"date": str(day), "symbol": symbol, "reason": "t_plus_one"}
                        )
                    quantity = min(quantity, sellable)
                    remaining = quantity
                    updated = []
                    for bought_on, old_quantity in self.lots[symbol]:
                        used = min(old_quantity, remaining) if bought_on < day else 0.0
                        remaining -= used
                        if old_quantity - used > 1e-10:
                            updated.append((bought_on, old_quantity - used))
                    # 清仓后移除记录，避免对已不持有的标的反复登记缺失行情阻断。
                    if updated:
                        self.lots[symbol] = updated
                    else:
                        del self.lots[symbol]
                    self.cash += quantity * price * (1 - self.commission - duty)
                else:
                    quantity = min(quantity, self.cash / (price * (1 + self.commission)))
                    self.cash -= quantity * price * (1 + self.commission)
                    self.lots.setdefault(symbol, []).append((day, quantity))
                if quantity > 1e-10:
                    self.trades.append(
                        {
                            "date": str(day),
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "price": price,
                            "commission": quantity * price * self.commission,
                            "stamp_duty": quantity * price * duty,
                            "slippage_cost": quantity * abs(price - float(row.open)),
                        }
                    )
        if self.cash < -1e-6:
            raise RuntimeError("现金不变量被破坏")
