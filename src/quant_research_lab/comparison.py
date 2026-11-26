"""在相同数据、选参、风控和成本下比较逐时点模型委员会与规则委员会。"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .agents import StructuredModel
from .analytics import relative_performance
from .data import MarketDataset, market_time
from .model_cache import CachedModel
from .models import ResearchConfig
from .reporting import save_report
from .research import run_research
from .store import AuditStore


def planned_model_calls(dataset: MarketDataset, config: ResearchConfig) -> int:
    count = len(dataset.dates) - config.training_sessions
    if count <= 1:
        raise ValueError("数据不足以形成样本外回放")
    folds = (count + config.test_sessions - 1) // config.test_sessions
    return 4 * (count + folds)


def paired_increment(
    llm: list[float], rules: list[float], seed: int, samples: int = 1999, block: int = 5
) -> dict[str, Any]:
    relative = relative_performance(llm, rules)
    a, b = np.asarray(llm), np.asarray(rules)
    active = a[1:] / a[:-1] - b[1:] / b[:-1]
    result: dict[str, Any] = {
        **relative,
        "sessions": len(active),
        "block_sessions": block,
        "seed": seed,
        "samples": samples,
        "method": "paired_moving_block_bootstrap_descriptive",
    }
    if len(active) < max(25, 5 * block):
        return {
            **result,
            "status": "insufficient_evidence",
            "daily_mean_interval": None,
            "reason": "少于五个区块；不把短冒烟测试当成模型增量证据",
        }
    rng = np.random.default_rng(seed)
    means = []
    blocks = (len(active) + block - 1) // block
    for _ in range(samples):
        starts = rng.integers(0, len(active) - block + 1, size=blocks)
        draws = np.concatenate([active[start : start + block] for start in starts])[: len(active)]
        means.append(float(draws.mean()))
    return {
        **result,
        "status": "descriptive_only",
        "daily_mean_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "reason": "单条市场路径的配对描述区间；未校正跨模型、跨实验选择，不是因果或盈利证明",
    }


def compare_research(
    dataset: MarketDataset,
    config: ResearchConfig,
    model: StructuredModel,
    provenance: dict[str, Any],
    output: Path,
    *,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    expected = planned_model_calls(dataset, config)
    budget = getattr(model, "max_calls", config.max_model_calls)
    if budget < expected:
        raise ValueError(
            f"完整模型回放预计{expected}次调用，预算仅{budget}；请显式提高max_model_calls"
        )
    released = provenance.get("released_at")
    if released:
        release = datetime.fromisoformat(released)
        if release.tzinfo is None:
            raise ValueError("模型发布日期必须带时区")
        first_decision = market_time(dataset.dates[config.training_sessions - 1])
        if first_decision <= release:
            raise ValueError("首个模型决策早于模型发布；请选择发布后的样本外数据")
    if output.exists() and any(output.iterdir()):
        raise ValueError("对照输出目录非空，不覆盖旧研究")
    output.mkdir(parents=True, exist_ok=True)
    registration: dict[str, Any] = {
        "dataset_sha256": dataset.digest,
        "config": config.model_dump(mode="json"),
        "model": provenance,
        "expected_model_calls": expected,
        "prompt_version": "committee_replay_v4",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "evidence_class": "synthetic_engineering_only"
        if dataset.manifest.kind == "synthetic"
        else "post_release_historical_replay"
        if released
        else "historical_contamination_uncontrolled",
    }
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    (output / "registration.json").write_text(
        json.dumps(registration, ensure_ascii=False, indent=2)
    )
    if model_cache is not None:
        if not provenance.get("digest"):
            raise ValueError("Resumable model responses require a pinned model digest")
        model = CachedModel(model, model_cache, registration)
    rules_dir, llm_dir = output / "rules", output / "llm"
    rules = run_research(dataset, config, AuditStore(rules_dir / "audit.sqlite"))
    save_report(rules, dataset.frame, rules_dir)
    try:
        llm = run_research(
            dataset,
            config,
            AuditStore(llm_dir / "audit.sqlite"),
            model=model,
            model_provenance=provenance,
        )
        save_report(llm, dataset.frame, llm_dir)
    except Exception as error:
        (output / "failed.json").write_text(
            json.dumps(
                {
                    "status": "model_replay_failed",
                    "error_type": type(error).__name__,
                    "completed_pair": False,
                },
                indent=2,
            )
        )
        raise
    a = [1.0, *[row["nav"] for row in llm["curve"]]]
    b = [1.0, *[row["nav"] for row in rules["curve"]]]
    increment = paired_increment(a, b, config.seed)
    windows_equal = [f["selected_window"] for f in rules["folds"]] == [
        f["selected_window"] for f in llm["folds"]
    ]
    if not windows_equal:
        raise RuntimeError("两组训练选参不一致，拒绝导出公平比较结论")
    decision_matches = sum(
        x["approved_weights"] == y["approved_weights"]
        for x, y in zip(rules["decision_log"], llm["decision_log"], strict=True)
    )
    result = {
        "registration": registration,
        "rules_run_id": rules["run_id"],
        "llm_run_id": llm["run_id"],
        "rules_metrics": rules["metrics"],
        "llm_metrics": llm["metrics"],
        "rules_evidence": rules["evidence"],
        "llm_evidence": llm["evidence"],
        "increment": increment,
        "training_windows_equal": windows_equal,
        "decision_match_fraction": decision_matches / len(llm["decision_log"]),
        "model_meetings": llm["model_meetings"],
        "model_veto_fraction": sum(bool(d["risk_violations"]) for d in llm["decision_log"])
        / len(llm["decision_log"]),
        "runtime_source_matches_registration": all(
            hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == value
            for name, value in registration["code_sha256"].items()
        ),
        "actual_model_calls": getattr(model, "calls", expected),
        "model_cache_hits": getattr(model, "cache_hits", 0),
        "model_cache_namespace": getattr(model, "namespace", None),
        "note": "模型负责逐时点仓位提议；预测命中率等旧指标仍是规则动量诊断，不是模型预测评分。",
        "disclaimer": "仅供研究，不构成投资建议",
    }
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    )
    from .comparison_report import render_comparison

    (output / "comparison.html").write_text(render_comparison(result), encoding="utf-8")
    return result
