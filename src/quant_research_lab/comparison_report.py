"""模型增量报告，分别展示模型、规则和证据边界。"""

import html
import json
from typing import Any

from .reporting import pct, table


def render_comparison(result: dict[str, Any]) -> str:
    rows = []
    for key, title in [("rules", "规则委员会"), ("llm", "逐时点大模型委员会")]:
        metrics, evidence = result[f"{key}_metrics"], result[f"{key}_evidence"]
        rows.append(
            {
                "后端": title,
                "累计收益": pct(metrics["total_return"]),
                "最大回撤": pct(metrics["max_drawdown"]),
                "相对等权基线": pct(evidence["relative_to_baseline"]["excess_total_return"]),
                "成本占比": pct(evidence["trading"]["total_cost_fraction"]),
                "成交笔数": evidence["trading"]["trade_count"],
                "熔断交易日": evidence["circuit_sessions"],
            }
        )
    increment = result["increment"]
    model = result["registration"]["model"]
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模型与规则委员会 · 对照研究</title>
<style>body{{background:#f4f7fa;color:#17243c;font:15px/1.8 system-ui,sans-serif;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 25px}}header{{border-top:5px solid #0f766e;padding:25px 0}}h1{{font-size:34px}}section{{background:white;border:1px solid #dce4ec;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.notice{{border-left:4px solid #d69e2e;padding:16px;background:#fff8e8}}</style><main><header><small>QUANT RESEARCH LAB / LLM vs RULES</small><h1>大模型每一次决策，都进入同一套回放。</h1><p>模型 {html.escape(str(model.get("name", "test-double")))} · {result["actual_model_calls"]} 次角色调用 · {result["model_meetings"]} 次会议</p></header>
<p class="notice">证据类型：{html.escape(result["registration"]["evidence_class"])}。短人工样本只验证真实模型接入和交易链路，不能判断投资效果。风控拒绝造成的现金收益不能解释为模型主动避险。</p>
<section><h2>相同数据、成本与风控</h2>{table(rows)}<p>训练窗口一致：{result["training_windows_equal"]}；获批仓位一致比例：{pct(result["decision_match_fraction"])}；模型提议否决比例：{pct(result.get("model_veto_fraction"))}。</p></section>
<section><h2>模型相对规则的增量</h2><p>累计收益差：{pct(increment["excess_total_return"])}；证据状态：{html.escape(increment["status"])}。</p><p>{html.escape(increment["reason"])}</p><pre>{html.escape(json.dumps(increment, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>证据强度怎么读</h2><p>保留两组原有证据指标；PSR不是模型带来增量的概率。增量比较使用相同日期的配对主动收益。{html.escape(result["note"])}</p><p>发布后历史回放与未来预注册测试分开；发布时间声明不证明模型从未接触过该段数据。</p><p><a href="rules/report.html">规则完整报告</a> · <a href="llm/report.html">模型完整报告</a></p></section><footer>仅供研究，不构成投资建议。</footer></main></html>"""
