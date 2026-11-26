"""可重复运行的离线研究命令；输入与输出禁止互相覆盖。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .agents import Committee, OllamaModel
from .comparison import compare_research
from .data import MarketDataset, market_time, synthetic_dataset
from .data_bundle import load_quant_bundle
from .models import ResearchConfig
from .openrouter import OpenRouterModel
from .prospective import create_commitment, load_commitment, prospective_step, settle_prospective
from .reporting import save_report
from .research import run_research
from .risk import RiskState
from .store import AuditStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant Research Lab；仅供研究，不构成投资建议")
    parser.add_argument(
        "command",
        choices=["demo", "research", "compare", "forward-init", "forward-step", "forward-settle"],
    )
    parser.add_argument("--bundle", type=Path, help="Hash-checked shared research bundle")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    parser.add_argument(
        "--ollama", action="store_true", help="仅增加最后时点的本地模型会议；不改变历史回测"
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        help="Shared validated response cache; rerun into a NEW output directory",
    )
    parser.add_argument("--model", help="Ollama 本机模型名，或 OpenRouter 模型ID")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openrouter"],
        default="ollama",
        help="compare 的模型后端；openrouter 需要 OPENROUTER_API_KEY 环境变量",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=1.0, help="OpenRouter 费用上限（美元），超出即停止"
    )
    parser.add_argument(
        "--openrouter-provider",
        help="固定单一服务商或端点标签，如 openai/flex；不设置则由 OpenRouter 路由",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="OpenRouter 推理强度；推理 token 按输出计费",
    )
    parser.add_argument("--sessions", type=int, default=240)
    parser.add_argument("--start-date", default="2023-01-02")
    parser.add_argument("--model-release", help="已核对模型版本的发布日期，如2024-10-24")
    parser.add_argument("--release-source", help="模型发布声明的一手来源链接")
    parser.add_argument("--model-timeout", type=float, default=60)
    parser.add_argument("--universe", nargs="+")
    parser.add_argument(
        "--calendar", type=Path, help="含decision_at/execution_at/mark_at的登记日历JSON"
    )
    args = parser.parse_args(argv)
    print("仅供研究，不构成投资建议")
    try:
        if args.bundle and args.command not in {"research", "compare"}:
            raise ValueError("--bundle supports research and compare only")
        if args.bundle and (args.input or args.manifest):
            raise ValueError("--bundle cannot be combined with --input or --manifest")
        config = (
            ResearchConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
            if args.config
            else ResearchConfig()
        )
        if args.command in {"forward-step", "forward-settle"}:
            if args.input is None or args.manifest is None:
                raise ValueError("前瞻步骤需要 --input 与 --manifest")
            dataset = MarketDataset.read(args.input, args.manifest)
            if args.command == "forward-settle":
                result = settle_prospective(args.output, dataset)
            else:
                commitment = load_commitment(args.output)
                model = OllamaModel(
                    model=commitment["model"]["name"],
                    seed=commitment["config"]["seed"],
                    max_calls=4,
                    expected_digest=commitment["model"]["digest"],
                    timeout=args.model_timeout,
                    audit_path=args.output / "model_calls.jsonl",
                )
                result = prospective_step(args.output, dataset, model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"compare", "forward-init"}:
            if not args.model_release or not args.release_source:
                raise ValueError(
                    "需要 --model-release 和 --release-source；不能用安装日期代替发布日"
                )
            replay: OllamaModel | OpenRouterModel
            if args.backend == "openrouter":
                if args.command == "forward-init":
                    raise ValueError("前瞻登记目前只支持本机 Ollama（需要可锁定的权重digest）")
                replay = OpenRouterModel(
                    model=args.model or "",
                    max_calls=config.max_model_calls,
                    seed=config.seed,
                    max_cost_usd=args.max_cost_usd,
                    audit_path=args.output / "model_calls.jsonl",
                    provider=args.openrouter_provider,
                    reasoning_effort=args.reasoning_effort,
                )
            else:
                replay = OllamaModel(
                    model=args.model,
                    max_calls=config.max_model_calls,
                    seed=config.seed,
                    timeout=args.model_timeout,
                    audit_path=args.output / "model_calls.jsonl",
                )
            metadata = replay.metadata()
            replay.expected_digest = metadata["digest"]
            provenance = {
                "name": metadata["name"],
                "digest": metadata["digest"],
                "details": metadata.get("details", {}),
                "backend": args.backend,
                "provider_routing": args.openrouter_provider,
                "reasoning_effort": args.reasoning_effort if args.backend == "openrouter" else None,
                "seed": config.seed,
                "released_at": datetime.fromisoformat(args.model_release)
                .replace(tzinfo=UTC)
                .isoformat(),
                "release_source": args.release_source,
                "release_date_status": "operator_declared_with_source",
            }
            if args.command == "forward-init":
                if not args.universe or args.calendar is None:
                    raise ValueError("前瞻登记必须显式提供 --universe 和 --calendar")
                result = create_commitment(
                    args.output,
                    config,
                    provenance,
                    tuple(args.universe),
                    json.loads(args.calendar.read_text()),
                )
                print(f"前瞻登记完成：{result['commitment_sha256']}；未来收益尚未发生。")
                return 0
            if args.bundle:
                dataset = load_quant_bundle(args.bundle)
            elif args.input is None:
                dataset = synthetic_dataset(config.seed, args.sessions, args.start_date)
            else:
                if args.manifest is None:
                    raise ValueError("用户数据需要manifest")
                dataset = MarketDataset.read(args.input, args.manifest)
            result = compare_research(
                dataset, config, replay, provenance, args.output, model_cache=args.model_cache
            )
            if isinstance(replay, OpenRouterModel):
                print(f"OpenRouter 实际费用 ${replay.cost_usd:.4f}")
            print(
                f"模型对照完成：{result['actual_model_calls']}次调用；"
                f"增量证据 {result['increment']['status']}；"
                f"报告 {args.output / 'comparison.html'}"
            )
            return 0
        if args.command == "demo":
            dataset = synthetic_dataset(config.seed, args.sessions, args.start_date)
        elif args.bundle:
            dataset = load_quant_bundle(args.bundle)
        else:
            if args.input is None or args.manifest is None:
                raise ValueError("research 必须同时提供 --input 与 --manifest")
            dataset = MarketDataset.read(args.input, args.manifest)
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("输出目录非空，请换一个目录，避免覆盖已有实验")
        args.output.mkdir(parents=True, exist_ok=True)
        store = AuditStore(args.output / "audit.sqlite")
        result = run_research(dataset, config, store)
        if args.ollama:
            model = OllamaModel(model=args.model, max_calls=config.max_model_calls)
            config_dump = result["curve"]
            nav = config_dump[-1]["nav"]
            previous = config_dump[-2]["nav"]
            state = RiskState(
                dataset.symbols,
                1 - nav / max(1.0, max(row["nav"] for row in config_dump)),
                nav / previous - 1,
            )
            view = dataset.view(market_time(dataset.dates[-1]))
            result["model_committee"] = Committee(config, model).decide(
                view, result["folds"][-1]["selected_window"], state, len(config.windows)
            )
            result["model_audit"] = model.audit
            result["model_scope"] = (
                "最后时点会议；未纳入历史回测；接口单测通过不等于真实模型效果已验证"
            )
        save_report(result, dataset.frame, args.output)
        print(
            f"研究完成：{result['run_id']}；{len(result['folds'])}个滚动测试段；"
            f"{result['parameter_trials']}次候选评估；报告：{args.output / 'report.html'}"
        )
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"运行停止：{type(error).__name__}。{str(error)[:300]}")
        print("检查输入字段/时区、配置与输出目录；本地模型需已启动且存在指定模型。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
