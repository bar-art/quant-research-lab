"""自包含报告：无远程资源，所有用户/模型文本做HTML转义。"""

import hashlib
import html
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd


def svg_curve(curve: list[dict[str, Any]]) -> str:
    series = [[1.0, *[row[key] for row in curve]] for key in ["nav", "baseline_nav"]]
    lower = min(min(values) for values in series)
    upper = max(max(values) for values in series)
    span = max(upper - lower, 0.01)
    paths = []
    for values, color in zip(series, ["#0f766e", "#94a3b8"], strict=True):
        points = " ".join(
            f"{35 + i / max(len(values) - 1, 1) * 830:.1f},{220 - (value - lower) / span * 180:.1f}"
            for i, value in enumerate(values)
        )
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    return (
        '<svg role="img" aria-label="样本外净值与等权基线" viewBox="0 0 900 250">'
        + "".join(paths)
        + "</svg>"
    )


def table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">无记录</p>'
    return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)


def pct(value: float | None) -> str:
    return "无" if value is None else f"{value:.2%}"


def num(value: float | None, digits: int = 2) -> str:
    return "无" if value is None else f"{value:.{digits}f}"


def evidence_rows(evidence: dict[str, Any]) -> list[dict[str, str]]:
    relative, trading, forecasts = (
        evidence["relative_to_baseline"],
        evidence["trading"],
        evidence["forecasts"],
    )
    return [
        {
            "指标": "相对基线超额累计收益",
            "数值": pct(relative["excess_total_return"]),
            "怎么读": "策略累计收益减等权基线累计收益",
        },
        {
            "指标": "信息比率（年化）",
            "数值": num(relative["information_ratio"]),
            "怎么读": "主动收益均值/跟踪误差；短样本噪声很大",
        },
        {
            "指标": "跑赢基线的测试段",
            "数值": f"{evidence['folds_beating_baseline']} / {evidence['fold_count']}",
            "怎么读": "逐段比较，看结果是否依赖少数区间",
        },
        {
            "指标": "PSR：夏普大于0的概率",
            "数值": pct(evidence["probabilistic_sharpe"]),
            "怎么读": "低于95%时不能认为夏普显著为正；未扣跨折选参偏差",
        },
        {
            "指标": "年化单边换手",
            "数值": f"{trading['annual_one_way_turnover']:.1f} 倍",
            "怎么读": f"基线 {evidence['baseline_trading']['annual_one_way_turnover']:.1f} 倍",
        },
        {
            "指标": "交易成本合计",
            "数值": pct(trading["total_cost_fraction"]),
            "怎么读": f"占起始资金；基线 {pct(evidence['baseline_trading']['total_cost_fraction'])}",
        },
        {
            "指标": "到期预测方向命中率",
            "数值": pct(forecasts["hit_rate"]),
            "怎么读": f"共 {int(forecasts['count'])} 条；50% 约等于随机",
        },
        {
            "指标": "预测秩相关 Rank IC 均值",
            "数值": num(forecasts["rank_ic_mean"], 3),
            "怎么读": f"按创建日横截面计算，{int(forecasts['ic_dates'])} 个日期",
        },
        {
            "指标": "风控熔断交易日",
            "数值": str(evidence["circuit_sessions"]),
            "怎么读": "熔断日的获批目标为空仓",
        },
    ]


def render_html(result: dict[str, Any]) -> str:
    escape = html.escape
    stats = result["metrics"]
    badge = (
        "人工行情 · 工程演示"
        if result["manifest"]["kind"] == "synthetic"
        else "用户数据 · 未独立核验"
    )
    cards = []
    for label, value in [
        ("样本外累计收益", f"{stats['total_return']:.2%}"),
        ("相对基线超额", pct(result["evidence"]["relative_to_baseline"]["excess_total_return"])),
        ("最大回撤", f"{stats['max_drawdown']:.2%}"),
        ("参数试验数", str(result["parameter_trials"])),
    ]:
        cards.append(f'<div class="card"><small>{label}</small><strong>{value}</strong></div>')
    fold_rows = [
        {
            "轮次": item["fold"],
            "训练结束": item["train_end"],
            "测试开始": item["test_start"],
            "测试结束": item["test_end"],
            "选中窗口": item["selected_window"],
            "训练收益": f"{next(t['metrics']['total_return'] for t in item['trials'] if t['window'] == item['selected_window']):.2%}",
            "样本外收益": f"{item['oos_metrics']['total_return']:.2%}",
            "基线收益": f"{item['baseline_oos_metrics']['total_return']:.2%}",
            "跑赢基线": "是" if item["beat_baseline"] else "否",
            "压力情景P05": f"{item['scenarios']['p05']:.2%}",
        }
        for item in result["folds"]
    ]
    committee = result.get("model_committee", result["final_committee"])
    notes = "".join(
        f'<article><small>{escape(note["role"])}</small><p>{escape(note["summary"])}</p><span class="muted">{escape(", ".join(note["evidence_ids"]))}</span></article>'
        for note in committee["notes"]
    )
    limitations = "".join(f"<li>{escape(text)}</li>" for text in result["limitations"])
    payload = escape(json.dumps(committee, ensure_ascii=False, indent=2))
    is_replay = result["backtest_backend"] == "llm_committee_replay"
    notice = (
        "本报告使用逐时点大模型提议，经过独立硬风控与模拟撮合；表现不代表稳定投资能力。"
        if is_replay
        else "本报告为规则策略回放；可选末期模型会议不参与其历史绩效。"
    )
    strategy_label = "模型委员会" if is_replay else "规则策略"
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant Research Lab · 研究报告</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:#17243c;font:15px/1.65 system-ui,"PingFang SC",sans-serif}}
main{{max-width:1160px;margin:auto;padding:48px 30px}}header{{border-top:5px solid #0f766e;padding:26px 0}}
h1{{font-size:38px;letter-spacing:-1px;margin:8px 0}}h2{{font-size:23px;margin-top:38px}}small{{text-transform:uppercase;letter-spacing:1px;color:#64748b}}
.badge{{display:inline-block;background:#fff1da;color:#86530b;border-radius:30px;padding:5px 14px;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.card,section,article{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:22px}}
.card strong{{display:block;font-size:29px;margin-top:10px}}section{{margin-top:18px;overflow:auto}}article{{margin:10px 0}}.muted{{color:#64748b;font-size:13px}}
svg{{width:100%;min-width:360px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:12px 10px;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap}}
th{{color:#64748b;background:#f8fafc}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.notice{{border-left:4px solid #eab308;padding-left:16px}}
footer{{border-top:1px solid #cbd5e1;margin-top:36px;padding-top:20px;color:#64748b}}@media(max-width:720px){{.grid{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:29px}}main{{padding:20px}}}}
</style><main><header><small>QUANT RESEARCH LAB / REPLAYABLE EVIDENCE</small><h1>从研究假设，到可检验的证据。</h1>
<span class="badge">{badge}</span><p class="muted">研究ID {escape(result["run_id"])} · 时点隔离 / 滚动检验 / 独立风控</p></header>
<p class="notice">{notice}</p>
<div class="grid">{"".join(cards)}</div><h2>样本外净值</h2><section>{svg_curve(result["curve"])}<p class="muted">绿色：{strategy_label}；灰色：相同总目标仓位的等权基线。起始净值1，已扣配置中的研究手续费与滑点。</p></section>
<h2>训练与测试分开报告</h2><section>{table(fold_rows)}<p class="muted">{escape(result["multiple_testing_warning"])} 压力情景只从训练收益抽样，不是预测分布。</p></section>
<h2>证据强度</h2><section>{table(evidence_rows(result["evidence"]))}<p class="muted">{escape(result["evidence"]["probabilistic_sharpe_note"])}</p>{table([{"熔断日期": e["date"], "原因": ", ".join(e["reasons"])} for e in result["evidence"]["circuit_events"][:12]]) if result["evidence"]["circuit_events"] else ""}</section>
<h2>四角色研究委员会</h2><p class="muted">后端：{escape(committee["backend"])}；三位分析师 + 一位经理；Python硬风控独立复核。</p>{notes}
<section><b>经理提议</b><p>{escape(committee["manager"]["summary"])}</p><p>获批目标：{escape(json.dumps(committee["approved_weights"]))}</p><p>风控阻断：{escape(", ".join(committee["risk_violations"]) or "无")}</p></section>
<h2>模拟成交与规则信号诊断</h2><p class="muted">下列预测评分来自规则动量信号，不是大模型预测的命中率。</p><section><p>{len(result["trades"])} 笔成交 · {len(result["blocked_orders"])} 条阻断 · {len(result["forecast_scores"])} 条到期评分</p>{table(result["forecast_scores"][:12])}</section>
<h2>数据与适用范围</h2><section><p>来源：{escape(result["manifest"]["source"])}</p><p>历史股票池完整声明：{result["manifest"]["historical_universe_complete"]}；公司行动完整声明：{result["manifest"]["corporate_actions_complete"]}</p><ul>{limitations}</ul></section>
<details><summary>查看最后一次结构化会议、证据与风控结果</summary><pre>{payload}</pre></details>
<footer>仅供研究，不构成投资建议。数据哈希 {escape(result["dataset_sha256"])}<br>参考：<a href="https://github.com/yebof/quant-agent">yebof/quant-agent</a> · <a href="https://arxiv.org/abs/2510.04643">QuantAgents 论文</a></footer></main></html>"""


def save_report(result: dict[str, Any], dataset_frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.html").write_text(render_html(result), encoding="utf-8")
    (output / "research.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name in ["curve", "trades", "blocked_orders", "forecast_scores"]:
        pd.DataFrame(result[name]).to_csv(output / f"{name}.csv", index=False)
    dataset_frame.to_csv(output / "input_bars.csv", index=False)
    (output / "input_manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_root = Path(__file__).parent
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.glob("*.py"))
    }
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "reproducibility.json"
    }
    import importlib.metadata

    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "pydantic"]}
    manifest = {
        "run_id": result["run_id"],
        "python": platform.python_version(),
        "versions": versions,
        "source_sha256": source_hashes,
        "artifacts_sha256": files,
        "seed": result["config"]["seed"],
        "execution_model": result["manifest"]["execution_model"],
    }
    (output / "reproducibility.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
