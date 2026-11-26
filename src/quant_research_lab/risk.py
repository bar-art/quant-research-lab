"""独立硬风控：模型只有提议权，违规提议全部拒绝。"""

import math
from dataclasses import dataclass

from .models import Proposal, RiskPolicy


@dataclass(frozen=True)
class RiskState:
    universe: tuple[str, ...]
    drawdown: float = 0.0
    daily_return: float = 0.0


class RiskEngine:
    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def review(self, proposal: Proposal, state: RiskState) -> tuple[dict[str, float], list[str]]:
        errors = []
        if not all(math.isfinite(v) for v in [state.drawdown, state.daily_return]):
            errors.append("nonfinite_state")
        if state.drawdown >= self.policy.max_drawdown:
            errors.append("drawdown_circuit")
        if state.daily_return <= -self.policy.max_daily_loss:
            errors.append("daily_loss_circuit")
        weights = proposal.weights
        for symbol, weight in weights.items():
            if symbol not in state.universe:
                errors.append(f"unknown_symbol:{symbol}")
            if not math.isfinite(weight) or weight < 0:
                errors.append(f"invalid_weight:{symbol}")
            elif weight > self.policy.max_position + 1e-10:
                errors.append(f"position_cap:{symbol}")
        gross = sum(weights.values())
        if not math.isfinite(gross) or gross > self.policy.max_gross + 1e-10:
            errors.append("gross_cap")
        if 1 - gross < self.policy.min_cash - 1e-10:
            errors.append("cash_floor")
        return ({} if errors else dict(weights)), errors
