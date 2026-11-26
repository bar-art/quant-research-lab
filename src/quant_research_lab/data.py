"""时间隔离的数据视图。执行层读取当日开盘，分析层只读取截止时点已知信息。"""

import hashlib
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .models import DataManifest, Evidence

TZ = ZoneInfo("Asia/Shanghai")


def market_time(day: date, hour: int = 15) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=TZ)


class FutureDataError(ValueError):
    pass


class AsOfView:
    """截止时点视图。由 MarketDataset.view() 创建时使用其逐标的索引（不复制全表）；
    直接用 DataFrame 构造时走逐行过滤的参考实现，两者结果逐项一致（见 test_view_index）。"""

    def __init__(
        self,
        frame: pd.DataFrame,
        cutoff: datetime,
        sessions: pd.DatetimeIndex | None = None,
        index: "MarketDataset | None" = None,
    ) -> None:
        if cutoff.tzinfo is None:
            raise ValueError("截止时点必须含时区")
        self.cutoff = cutoff
        self.__sessions = (
            sessions
            if sessions is not None
            else pd.DatetimeIndex(frame.trade_date.unique()).sort_values()
        )
        self.__source = frame
        self.__index = index
        self.__visible_frame: pd.DataFrame | None = None
        self.__visible_rows: dict[str, np.ndarray] | None = None
        self.__latest_rows: (
            list[tuple[str, pd.Timestamp, pd.Timestamp, float, bool, int]] | None
        ) = None
        self.__momentum: dict[int, dict[str, float]] = {}

    @property
    def __frame(self) -> pd.DataFrame:
        # 只有 panel() 与参考实现需要完整可见表；按需构建一次。
        if self.__visible_frame is None:
            self.__visible_frame = self.__source[self.__source.available_at <= self.cutoff].copy()
        return self.__visible_frame

    def __rows(self) -> dict[str, np.ndarray]:
        """每个标的截止时点已可得的行位置（按交易日排序）；发布延迟可以不单调。"""
        if self.__visible_rows is None:
            assert self.__index is not None
            self.__visible_rows = self.__index.visible_rows(self.cutoff)
        return self.__visible_rows

    def momentum(self, window: int) -> dict[str, float]:
        """features(panel(), window) 的缓存版本；同一会议中多个角色共用，结果与逐次计算一致。"""
        if window not in self.__momentum:
            if self.__index is not None:
                self.__momentum[window] = self.__index.scores(self.cutoff, window)
            else:
                from .analytics import features

                self.__momentum[window] = features(self.__frame, window)
        return dict(self.__momentum[window])

    def history(self, symbol: str, end: date | None = None) -> pd.DataFrame:
        if end is not None and end > self.cutoff.astimezone(TZ).date():
            raise FutureDataError("禁止读取截止时点后的数据")
        if self.__index is not None:
            rows = self.__rows().get(symbol)
            group = self.__index.symbol_frame(symbol)
            selected = group.iloc[rows] if rows is not None else group.iloc[:0]
        else:
            selected = self.__frame[self.__frame.ts_code == symbol]
        if end is not None:
            selected = selected[selected.trade_date <= pd.Timestamp(end)]
        return selected.sort_values("trade_date").copy()

    def signal_age(self, symbol: str) -> int | None:
        rows = self.history(symbol)
        if rows.empty:
            return None
        last = rows.iloc[-1].trade_date
        current = pd.Timestamp(self.cutoff.astimezone(TZ).date())
        if current not in self.__sessions:
            return None
        return int(((self.__sessions > last) & (self.__sessions <= current)).sum())

    def __latest(self) -> list[tuple[str, pd.Timestamp, pd.Timestamp, float, bool, int]]:
        """每个标的最后一条可见行（交易日、可得时点、收盘、资格）与可见行数，按代码排序。"""
        if self.__latest_rows is None:
            assert self.__index is not None
            result = []
            for symbol, rows in sorted(self.__rows().items()):
                days, available, closes, _volumes, _suspended, members = self.__index.series(symbol)
                last = int(rows[-1])
                result.append(
                    (
                        symbol,
                        pd.Timestamp(days[last]),
                        pd.Timestamp(int(available[last]), tz="UTC"),
                        float(closes[last]),
                        bool(members[last]),
                        len(rows),
                    )
                )
            self.__latest_rows = result
        return self.__latest_rows

    def universe(self) -> tuple[str, ...]:
        if self.__index is not None:
            return tuple(item[0] for item in self.__latest() if item[4])
        latest = self.__frame.sort_values("trade_date").groupby("ts_code").tail(1)
        return tuple(sorted(latest.loc[latest.universe_member, "ts_code"].astype(str)))

    def panel(self) -> pd.DataFrame:
        return self.__frame.copy()

    @staticmethod
    def __note(
        symbol: str, trade_date: pd.Timestamp, available_at: Any, close: float, count: int
    ) -> Evidence:
        return Evidence(
            id=f"price:{symbol}:{trade_date.date()}",
            available_at=pd.Timestamp(available_at).to_pydatetime(),
            summary=f"{symbol} 已知收盘价 {close:.4f}，{count}条历史",
        )

    def evidence(self) -> tuple[Evidence, ...]:
        if self.__index is not None:
            return tuple(
                self.__note(symbol, day, stamp, close, count)
                for symbol, day, stamp, close, _member, count in self.__latest()
            )
        notes = []
        # 一次分组代替逐个标的过滤排序；每组按交易日排序后取最后一行，与 history() 一致。
        ordered = self.__frame.sort_values("trade_date")
        for symbol, rows in ordered.groupby("ts_code", sort=True):
            latest = rows.iloc[-1]
            notes.append(
                self.__note(
                    str(symbol), latest.trade_date, latest.available_at, latest.close, len(rows)
                )
            )
        return tuple(notes)


class MarketDataset:
    def __init__(self, frame: pd.DataFrame, manifest: DataManifest) -> None:
        required = {
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "available_at",
            "is_suspended",
            "up_limit",
            "down_limit",
            "price_limit_exempt",
        }
        if required - set(frame):
            raise ValueError(f"输入缺少字段：{sorted(required - set(frame))}")
        data = frame.copy()
        if "universe_member" not in data:
            data["universe_member"] = True
        data["trade_date"] = pd.to_datetime(data.trade_date, errors="raise").dt.normalize()
        # 拒绝无时区输入，不能把本地时间静默当作UTC。
        stamps = [pd.Timestamp(value) for value in data.available_at]
        if any(value.tzinfo is None for value in stamps):
            raise ValueError("available_at 必须带时区")
        data["available_at"] = pd.to_datetime(data.available_at, utc=True)
        if data.empty or data.trade_date.isna().any() or data.available_at.isna().any():
            raise ValueError("行情或时间戳为空")
        if data.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("行情键重复；请在导入前按可得时点处理更正版本")
        if not data.ts_code.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("标的代码必须为非空字符串")
        for column in ["open", "high", "low", "close", "volume"]:
            values = pd.to_numeric(data[column], errors="raise").astype(float)
            if (
                not np.isfinite(values).all()
                or (values < (0 if column == "volume" else 1e-12)).any()
            ):
                raise ValueError(f"{column} 包含非法数值")
            data[column] = values
        if (
            (data.high < data[["open", "close", "low"]].max(axis=1))
            | (data.low > data[["open", "close", "high"]].min(axis=1))
        ).any():
            raise ValueError("高低价不能覆盖开收盘价")
        for column in ["is_suspended", "price_limit_exempt", "universe_member"]:
            if not data[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"{column} 必须显式提供布尔值，不能把未知状态当作否")
        for column in ["up_limit", "down_limit"]:
            data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
            known = data[column].dropna()
            if not np.isfinite(known).all() or (known <= 0).any():
                raise ValueError("涨跌停价格须为正数或明确缺失")
        minimum = pd.to_datetime([market_time(day.date()) for day in data.trade_date], utc=True)
        if (data.available_at.to_numpy() < minimum.to_numpy()).any():
            raise ValueError("完整日线的 available_at 不能早于当日收盘")
        self.frame = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        # Immutable lookup snapshots: no full-panel mask/copy in the training hot loop.
        self._daily = {
            pd.Timestamp(str(day)).date(): group.copy()
            for day, group in self.frame.groupby("trade_date")
        }
        self._series = {
            str(symbol): (
                group.trade_date.to_numpy(),
                group.available_at.dt.as_unit("ns").astype("int64").to_numpy(),
                group.close.to_numpy(),
                group.volume.to_numpy(),
                group.is_suspended.to_numpy(),
                group.universe_member.to_numpy(),
            )
            for symbol, group in self.frame.groupby("ts_code")
        }
        self._frames = {
            str(symbol): group.sort_values("trade_date", kind="stable")
            for symbol, group in self.frame.groupby("ts_code")
        }
        self._empty = self.frame.iloc[:0]
        self._symbol_names = np.array(sorted(str(name) for name in self.frame.ts_code.unique()))
        self._flat = {
            "day": self.frame.trade_date.to_numpy(),
            "available": self.frame.available_at.dt.as_unit("ns").astype("int64").to_numpy(),
            "close": self.frame.close.to_numpy(),
            "volume": self.frame.volume.to_numpy(),
            "suspended": self.frame.is_suspended.to_numpy(dtype=bool),
            "member": self.frame.universe_member.to_numpy(dtype=bool),
            "symbol": np.searchsorted(
                self._symbol_names, self.frame.ts_code.astype(str).to_numpy()
            ),
        }
        self.manifest = manifest
        self.dates = [stamp.date() for stamp in sorted(data.trade_date.unique())]
        self.symbols = tuple(sorted(str(symbol) for symbol in data.ts_code.unique()))
        self._session_index = pd.DatetimeIndex(sorted(data.trade_date.unique()))
        canonical = (
            self.frame.to_csv(index=False, float_format="%.12g") + manifest.model_dump_json()
        )
        self.digest = hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def read(cls, path: Path, manifest_path: Path) -> "MarketDataset":
        frame = pd.read_csv(path, dtype={"ts_code": str})
        return cls(
            frame, DataManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        )

    def view(self, cutoff: datetime) -> AsOfView:
        return AsOfView(self.frame, cutoff, self._session_index, index=self)

    def symbol_frame(self, symbol: str) -> pd.DataFrame:
        """单个标的全部行（按交易日排序，只读）；未知标的返回空表。"""
        found = self._frames.get(symbol)
        return found if found is not None else self._empty

    def series(self, symbol: str) -> tuple[np.ndarray, ...]:
        """单个标的按交易日排序的数组：交易日、可得时点(ns)、收盘、成交量、停牌、资格。"""
        return self._series[symbol]

    def visible_rows(self, cutoff: datetime) -> dict[str, np.ndarray]:
        """截止时点 available_at 已到达的行位置；只含至少有一行可见的标的。"""
        if cutoff.tzinfo is None:
            raise ValueError("截止时点必须含时区")
        stamp = pd.Timestamp(cutoff).value
        last_day = np.datetime64(cutoff.astimezone(TZ).date())
        result = {}
        for symbol, (days, available, *_rest) in self._series.items():
            # available_at 不早于当日收盘，交易日晚于截止日的行不可能可见。
            stop = int(np.searchsorted(days, last_day, side="right"))
            rows = np.flatnonzero(available[:stop] <= stamp)
            if len(rows):
                result[symbol] = rows
        return result

    def bars(self, day: date) -> pd.DataFrame:
        group = self._daily.get(day)
        return group.copy() if group is not None else self.frame.iloc[:0].copy()

    def scores(self, cutoff: datetime, window: int, start: date | None = None) -> dict[str, float]:
        """Same semantics as features(view.panel()), vectorised over the date-sorted frame.

        Publication delays need not be monotone; filter availability before selecting
        the lag. Never infer visibility from trade_date alone.
        """
        if cutoff.tzinfo is None:
            raise ValueError("截止时点必须含时区")
        flat = self._flat
        stamp = pd.Timestamp(cutoff).value
        # frame 按交易日排序：先按日期切片（训练段只需约 80 个交易日），再筛可得时点。
        lo = int(np.searchsorted(flat["day"], np.datetime64(start))) if start else 0
        hi = int(
            np.searchsorted(flat["day"], np.datetime64(cutoff.astimezone(TZ).date()), side="right")
        )
        rows = np.arange(lo, hi)[flat["available"][lo:hi] <= stamp]
        if not len(rows):
            return {}
        # 稳定排序按标的分组，组内保持交易日顺序。
        rows = rows[np.argsort(flat["symbol"][rows], kind="stable")]
        codes = flat["symbol"][rows]
        ends = np.flatnonzero(np.r_[codes[1:] != codes[:-1], True])
        counts = np.diff(np.r_[-1, ends])
        last = rows[ends]
        latest = flat["day"][last].max()
        ok = (
            (counts > window)
            & (flat["volume"][last] > 0)
            & ~flat["suspended"][last]
            & flat["member"][last]
            & (flat["day"][last] == latest)
        )
        base = rows[(ends - window)[ok]]
        values = flat["close"][last[ok]] / flat["close"][base] - 1
        names = self._symbol_names[codes[ends][ok]]
        return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def synthetic_dataset(
    seed: int = 42, sessions: int = 240, start_date: str = "2023-01-02"
) -> MarketDataset:
    if sessions < 100:
        raise ValueError("演示至少100个合成交易时点")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=sessions)
    rows: list[dict[str, Any]] = []
    for index, symbol in enumerate(["SYN_A", "SYN_B", "SYN_C", "SYN_D"]):
        previous = 100.0
        for i, day in enumerate(dates):
            drift = (0.001 if i < sessions // 2 else -0.0004) * (1 if index < 2 else -0.5)
            opening = previous * np.exp(rng.normal(0, 0.003))
            closing = opening * np.exp(rng.normal(drift, 0.012))
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": opening,
                    "close": closing,
                    "high": max(opening, closing) * 1.003,
                    "low": min(opening, closing) * 0.997,
                    "volume": 1000000.0,
                    "available_at": market_time(day.date()).isoformat(),
                    "is_suspended": False,
                    "price_limit_exempt": True,
                    "up_limit": np.nan,
                    "down_limit": np.nan,
                }
            )
            previous = closing
    manifest = DataManifest(
        kind="synthetic", source=f"人工随机样本 seed={seed}；工作日不是真实交易日历"
    )
    return MarketDataset(pd.DataFrame(rows), manifest)
