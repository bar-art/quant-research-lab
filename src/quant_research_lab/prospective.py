"""真实前瞻协议：先登记，按实际时钟记决策，到期才结算；不补写历史提议。"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import Committee, OllamaModel, StructuredModel
from .analytics import performance
from .comparison import paired_increment
from .data import TZ, MarketDataset, market_time
from .execution import PaperBook
from .models import ResearchConfig
from .research import new_book, select_window
from .risk import RiskState


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def code_hash() -> str:
    return hashlib.sha256(
        canonical(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ).encode()
    ).hexdigest()


def aware(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("前瞻时点必须包含时区")
    return result


def create_commitment(
    root: Path,
    config: ResearchConfig,
    provenance: dict[str, Any],
    universe: tuple[str, ...],
    calendar: list[dict[str, str]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or not universe or len(set(universe)) != len(universe):
        raise ValueError("时钟或股票池无效")
    if (
        not provenance.get("digest")
        or not provenance.get("released_at")
        or not provenance.get("release_source")
    ):
        raise ValueError("前瞻必须锁定模型digest、发布时间声明及来源")
    if aware(provenance["released_at"]) >= now:
        raise ValueError("模型还未发布，不能登记")
    if not calendar or len(calendar) * 4 > config.max_model_calls:
        raise ValueError("前瞻日历为空或模型调用预算不足")
    previous_mark: datetime | None = None
    for session in calendar:
        decision, execution, mark = [
            aware(session[key]) for key in ["decision_at", "execution_at", "mark_at"]
        ]
        if not now < decision < execution <= mark:
            raise ValueError("每个决策必须晚于登记且早于执行；执行后才可结算")
        if decision.astimezone(TZ) != market_time(decision.astimezone(TZ).date()):
            raise ValueError("当前协议固定为上海时间15:00收盘决策")
        if execution.astimezone(TZ).date() <= decision.astimezone(TZ).date():
            raise ValueError("执行必须在后续交易日，不能当日收盘成交")
        local_execution = execution.astimezone(TZ)
        if (local_execution.hour, local_execution.minute) != (9, 30):
            raise ValueError("当前协议固定为上海时间09:30开盘执行")
        if mark.astimezone(TZ) != market_time(mark.astimezone(TZ).date()):
            raise ValueError("当前协议固定为上海时间15:00收盘结算")
        if mark.astimezone(TZ).date() != execution.astimezone(TZ).date():
            raise ValueError("结算必须是执行日收盘")
        if previous_mark is not None and decision < previous_mark:
            raise ValueError("前一期必须先到期，再产生下一期提议")
        previous_mark = mark
    if root.exists() and any(root.iterdir()):
        raise ValueError("前瞻目录非空，不能重写登记")
    root.mkdir(parents=True, exist_ok=True)
    body = {
        "registered_at": now.isoformat(),
        "model": provenance,
        "universe": list(universe),
        "calendar": calendar,
        "config": config.model_dump(mode="json"),
        "code_hash": code_hash(),
        "protocol": "close_decision_next_session_open_v1",
        "timestamp_strength": "local_clock_only",
    }
    body["commitment_sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    (root / "commitment.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with sqlite3.connect(root / "prospective.sqlite") as db:
        db.executescript("""CREATE TABLE identity (digest TEXT);
          CREATE TABLE decisions (
            seq INTEGER PRIMARY KEY, status TEXT, payload TEXT, decided_at TEXT);
          CREATE TABLE settlements (seq INTEGER PRIMARY KEY, payload TEXT, settled_at TEXT);
          CREATE TABLE state (id INTEGER PRIMARY KEY, payload TEXT);""")
        db.execute("INSERT INTO identity VALUES (?)", [body["commitment_sha256"]])
        db.execute(
            "INSERT INTO state VALUES (1,?)",
            [
                canonical(
                    {
                        arm: {"cash": 100000.0, "lots": {}, "last_prices": {}, "values": [100000.0]}
                        for arm in ["rules", "llm"]
                    }
                )
            ],
        )
    return body


def load_commitment(root: Path) -> dict[str, Any]:
    body = json.loads((root / "commitment.json").read_text())
    claimed = body.pop("commitment_sha256")
    if hashlib.sha256(canonical(body).encode()).hexdigest() != claimed:
        raise ValueError("前瞻登记内容被修改")
    body["commitment_sha256"] = claimed
    if body["code_hash"] != code_hash():
        raise ValueError("前瞻登记后代码发生变化，请保留原版本运行，不补写旧研究")
    with sqlite3.connect(root / "prospective.sqlite") as db:
        if db.execute("SELECT digest FROM identity").fetchone() != (claimed,):
            raise ValueError("前瞻数据库与登记不匹配")
    return dict(body)


def restore_book(state: dict[str, Any], config: ResearchConfig) -> PaperBook:
    book = new_book(config)
    book.cash = state["cash"]
    book.last_prices = state["last_prices"]
    book.lots = {
        symbol: [(datetime.fromisoformat(day).date(), quantity) for day, quantity in lots]
        for symbol, lots in state["lots"].items()
    }
    return book


def book_state(book: PaperBook, values: list[float]) -> dict[str, Any]:
    return {
        "cash": book.cash,
        "lots": {
            symbol: [(str(day), quantity) for day, quantity in lots]
            for symbol, lots in book.lots.items()
        },
        "last_prices": book.last_prices,
        "values": values,
    }


def prospective_step(
    root: Path, dataset: MarketDataset, model: StructuredModel, *, now: datetime | None = None
) -> dict[str, Any]:
    supplied_now = now
    now = now or datetime.now(UTC)
    commitment = load_commitment(root)
    if dataset.manifest.kind == "synthetic":
        raise ValueError("真实前瞻禁止使用人工行情")
    config = ResearchConfig.model_validate(commitment["config"])
    if (
        isinstance(model, OllamaModel)
        and model.metadata()["digest"] != commitment["model"]["digest"]
    ):
        raise ValueError("模型digest与预注册不符")
    with sqlite3.connect(root / "prospective.sqlite") as db:
        db.execute("BEGIN IMMEDIATE")
        count = db.execute("SELECT count(*) FROM decisions").fetchone()[0]
        if count >= len(commitment["calendar"]):
            raise ValueError("全部计划决策已登记；不能重复请求以挑选输出")
        session = commitment["calendar"][count]
        cutoff, execution = aware(session["decision_at"]), aware(session["execution_at"])
        if not cutoff <= now < execution:
            raise ValueError("不在本期实时决策窗口；禁止事后补写或预先生成未来决策")
        if count and db.execute("SELECT count(*) FROM settlements").fetchone()[0] != count:
            raise ValueError("请先结算已到期的上一期，再产生下一期决策")
        states = json.loads(db.execute("SELECT payload FROM state WHERE id=1").fetchone()[0])
        db.execute("INSERT INTO decisions VALUES (?, 'started', NULL, ?)", [count, now.isoformat()])
    try:
        view = dataset.view(cutoff)
        visible = view.panel()
        if set(visible.ts_code.unique()) != set(commitment["universe"]):
            raise ValueError("输入股票池与预注册不一致")
        current_bars = visible[visible.trade_date == pd.Timestamp(cutoff.astimezone(TZ).date())]
        if set(current_bars.ts_code) != set(commitment["universe"]):
            raise ValueError("当期股票池行情未全部按时公开")
        dates = sorted(day.date() for day in visible.trade_date.unique())
        if len(dates) < config.training_sessions or dates[-1] != cutoff.astimezone(TZ).date():
            raise ValueError("没有截止本期的完整训练历史")
        selected, trials, scenario = select_window(
            dataset, dates[-config.training_sessions :], config, count
        )
        payload: dict[str, Any] = {
            "as_of": cutoff.isoformat(),
            "input_sha256": dataset.digest,
            "recorded_at": now.isoformat(),
            "selected_window": selected,
        }
        for arm, client in [("rules", None), ("llm", model)]:
            state = states[arm]
            book = restore_book(state, config)
            current = book.value(current_bars)
            history = state["values"]
            prior = (
                history[-2]
                if len(history) > 1 and state.get("marked_at") == cutoff.isoformat()
                else history[-1]
            )
            peak_from = (
                (count // config.test_sessions) * config.test_sessions
                if config.drawdown_reset == "next_fold"
                else 0
            )
            risk = RiskState(
                tuple(commitment["universe"]),
                1 - current / max(*history[peak_from:], current),
                current / prior - 1,
            )
            payload[arm] = Committee(config, client).decide(
                view,
                selected,
                risk,
                len(config.windows),
                {
                    "candidate_results": trials,
                    "scenarios": scenario,
                    "portfolio": {"cash_fraction": book.cash / current},
                },
            )
        completed_at = datetime.now(UTC) if supplied_now is None else supplied_now
        if completed_at >= execution:
            raise ValueError("模型决策完成时已超过执行时点，不能事后采用输出")
        with sqlite3.connect(root / "prospective.sqlite") as db:
            db.execute(
                "UPDATE decisions SET status='completed',payload=? WHERE seq=?",
                [canonical(payload), count],
            )
        return payload
    except Exception as error:
        with sqlite3.connect(root / "prospective.sqlite") as db:
            db.execute(
                "UPDATE decisions SET status='failed',payload=? WHERE seq=?",
                [canonical({"error_type": type(error).__name__}), count],
            )
        raise


def settle_prospective(
    root: Path, dataset: MarketDataset, *, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    commitment = load_commitment(root)
    if dataset.manifest.kind == "synthetic":
        raise ValueError("真实前瞻禁止使用人工行情结算")
    config = ResearchConfig.model_validate(commitment["config"])
    with sqlite3.connect(root / "prospective.sqlite") as db:
        db.execute("BEGIN IMMEDIATE")
        seq = db.execute("SELECT count(*) FROM settlements").fetchone()[0]
        if seq >= len(commitment["calendar"]):
            raise ValueError("全部计划期已结算，不重复评分")
        session = commitment["calendar"][seq]
        if now < aware(session["mark_at"]):
            raise ValueError("本期尚未到期，不得提前计算未来收益")
        row = db.execute("SELECT status,payload FROM decisions WHERE seq=?", [seq]).fetchone()
        if row is None or row[0] != "completed":
            raise ValueError("缺少按时完成的决策，不能补写前瞻记录")
        decisions = json.loads(row[1])
        states = json.loads(db.execute("SELECT payload FROM state WHERE id=1").fetchone()[0])
        day = aware(session["mark_at"]).astimezone(TZ).date()
        bars = dataset.bars(day)
        if set(bars.ts_code) != set(commitment["universe"]) or (bars.available_at > now).any():
            raise ValueError("本期结算行情缺失、股票池不匹配或尚未公开")
        payload: dict[str, Any] = {
            "seq": seq,
            "settled_at": now.isoformat(),
            "input_sha256": dataset.digest,
        }
        for arm in ["rules", "llm"]:
            state = states[arm]
            book = restore_book(state, config)
            book.rebalance(
                decisions[arm]["approved_weights"], bars, day, aware(session["decision_at"])
            )
            nav = book.mark_close(bars)
            states[arm] = {
                **book_state(book, state["values"] + [nav]),
                "marked_at": session["mark_at"],
            }
            payload[arm] = {"nav": nav, "trades": book.trades, "blocked_orders": book.blocks}
        db.execute(
            "INSERT INTO settlements VALUES (?,?,?)", [seq, canonical(payload), now.isoformat()]
        )
        db.execute("UPDATE state SET payload=? WHERE id=1", [canonical(states)])
    a, b = states["llm"]["values"], states["rules"]["values"]
    result = {
        "status": "complete" if seq + 1 == len(commitment["calendar"]) else "in_progress",
        "settled_sessions": seq + 1,
        "planned_sessions": len(commitment["calendar"]),
        "rules_metrics": performance(b),
        "llm_metrics": performance(a),
        "increment": paired_increment(a, b, config.seed),
        "commitment_sha256": commitment["commitment_sha256"],
        "note": "本地时钟/数据源自报；无第三方不可篡改时间戳或市场数据独立审计",
    }
    (root / "forward_grade.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
