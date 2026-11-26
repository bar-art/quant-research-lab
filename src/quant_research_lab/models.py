"""角色协议与不可由模型修改的研究配置。"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class RiskPolicy(Contract):
    max_position: float = Field(default=0.3, gt=0, le=1)
    max_gross: float = Field(default=0.6, gt=0, le=1)
    min_cash: float = Field(default=0.4, ge=0, le=1)
    max_drawdown: float = Field(default=0.15, gt=0, le=1)
    max_daily_loss: float = Field(default=0.04, gt=0, le=1)

    @model_validator(mode="after")
    def consistent(self) -> "RiskPolicy":
        if self.max_gross > 1 - self.min_cash + 1e-10:
            raise ValueError("总仓位上限必须留足最低现金")
        return self


class ResearchConfig(Contract):
    seed: int = Field(default=42, ge=0)
    training_sessions: int = Field(default=80, ge=30)
    test_sessions: int = Field(default=20, ge=2)
    windows: tuple[int, ...] = (5, 10, 20)
    top_k: int = Field(default=2, ge=1, le=20)
    commission_bps: float = Field(default=5, ge=0, le=100)
    slippage_bps: float = Field(default=2, ge=0, le=100)
    # 卖出单边税费，例如A股印花税（2023-08-28起为5bps）；默认0以保持统一研究费用口径。
    stamp_duty_bps: float = Field(default=0, ge=0, le=100)
    # 分段印花税（生效日, 卖出bps），A股见 config/ashare.json；不得与固定税率同时设置。
    stamp_duty_schedule: tuple[tuple[date, float], ...] = ()
    scenario_paths: int = Field(default=250, ge=20, le=10000)
    scenario_horizon: int = Field(default=10, ge=1, le=60)
    policy: RiskPolicy = RiskPolicy()
    # never：回撤熔断后不再开仓（保守）；next_fold：新测试段重新选参后以当时净值为新高点重新计数。
    drawdown_reset: Literal["never", "next_fold"] = "never"
    max_model_calls: int = Field(default=4, ge=1, le=100000)
    max_signal_age_sessions: int = Field(default=0, ge=0, le=5)

    @model_validator(mode="after")
    def valid_stamp_duty(self) -> "ResearchConfig":
        days = [day for day, _ in self.stamp_duty_schedule]
        if self.stamp_duty_schedule and self.stamp_duty_bps:
            raise ValueError("分段印花税与固定 stamp_duty_bps 只能设置一种")
        if days != sorted(set(days)):
            raise ValueError("印花税分段生效日必须严格递增")
        if any(not 0 <= bps <= 100 for _, bps in self.stamp_duty_schedule):
            raise ValueError("印花税 bps 须在 0 到 100 之间")
        return self

    @model_validator(mode="after")
    def valid_windows(self) -> "ResearchConfig":
        if not self.windows or len(set(self.windows)) != len(self.windows):
            raise ValueError("候选窗口必须非空且不重复")
        if min(self.windows) < 2 or max(self.windows) >= self.training_sessions // 2:
            raise ValueError("候选窗口需小于训练长度的一半且至少2日")
        return self


class DataManifest(Contract):
    kind: Literal["synthetic", "user_supplied"]
    source: str = Field(min_length=3)
    historical_universe_complete: bool = False
    corporate_actions_complete: bool = False
    execution_model: Literal["fractional_next_open_research"] = "fractional_next_open_research"
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"


class Evidence(Contract):
    id: str = Field(min_length=1)
    available_at: datetime
    summary: str = Field(min_length=1, max_length=500)

    @field_validator("available_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("证据时间戳必须含时区")
        return value


class AgentNote(Contract):
    role: Literal["market", "simulation", "risk", "manager"]
    summary: str = Field(min_length=1, max_length=1200)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=30)
    concerns: tuple[str, ...] = ()


class Proposal(Contract):
    weights: dict[str, float]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=1200)


class Forecast(Contract):
    symbol: str
    created_at: datetime
    due_at: datetime
    predicted_return: float

    @model_validator(mode="after")
    def order(self) -> "Forecast":
        for value in [self.created_at, self.due_at]:
            if value.tzinfo is None:
                raise ValueError("预测时间戳必须含时区")
        if self.due_at <= self.created_at:
            raise ValueError("预测到期时点必须晚于创建时点")
        return self
