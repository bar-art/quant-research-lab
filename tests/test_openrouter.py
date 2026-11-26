import json

import pandas as pd
import pytest

from quant_research_lab.data import MarketDataset
from quant_research_lab.models import AgentNote, Proposal
from quant_research_lab.openrouter import OpenRouterModel

CATALOG = {
    "data": [
        {
            "id": "openai/gpt-oss-120b",
            "canonical_slug": "openai/gpt-oss-120b",
            "created": 1754352000,
            "context_length": 131072,
            "pricing": {"prompt": "0.00000005"},
        }
    ]
}


def fake_transport(sent, content=None, cost=0.001):
    """按请求中的 schema 返回合法的委员会输出；记录发出的请求。"""

    def transport(path, body, api_key):
        if body is None:
            assert api_key == ""  # 公开目录查询不带密钥
            return CATALOG
        assert api_key == "test-key"
        sent.append(body)
        payload = json.loads(body["messages"][1]["content"])
        role = body["response_format"]["json_schema"]["name"]
        ids = [payload["evidence"][0].split("|")[0]]
        if content is not None:
            text = content
        elif role == "manager":
            text = json.dumps({"weights": [], "evidence_ids": ids, "summary": "stay in cash"})
        else:
            body_schema = body["response_format"]["json_schema"]
            assert body_schema["strict"] and body_schema["schema"]["additionalProperties"] is False
            text = json.dumps(
                {"role": role, "summary": "known data", "evidence_ids": ids, "concerns": []}
            )
        return {
            "model": body["model"],
            "provider": "UnitProvider",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": cost},
        }

    return transport


def payload(dataset):
    view = dataset.view(
        pd.Timestamp(dataset.dates[50]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    )
    evidence = [item.model_dump(mode="json") for item in view.evidence()]
    return {
        "evidence": evidence,
        "momentum": view.momentum(5),
        "policy": {"max_position": 0.3, "max_gross": 0.6},
        "top_k": 2,
    }


def test_generate_uses_shared_protocol_and_never_logs_key(dataset, tmp_path):
    sent: list[dict] = []
    model = OpenRouterModel(
        "openai/gpt-oss-120b",
        max_calls=3,
        audit_path=tmp_path / "calls.jsonl",
        api_key="test-key",
        transport=fake_transport(sent),
    )
    data = payload(dataset)
    note = model.generate("market", data, AgentNote)
    proposal = model.generate("manager", data, Proposal)
    assert note.role == "market" and proposal.weights == {}
    schema = sent[1]["response_format"]["json_schema"]["schema"]
    weight_item = schema["properties"]["weights"]["items"]
    assert weight_item["properties"]["symbol"]["enum"] == sorted(data["momentum"])
    # 紧凑证据：一行一条，信息与原摘要一致。
    first = json.loads(sent[0]["messages"][1]["content"])["evidence"][0]
    evidence_id, close, count = first.split("|")
    assert evidence_id == data["evidence"][0]["id"]
    assert f"已知收盘价 {close}，{count}条历史" in data["evidence"][0]["summary"]
    assert sent[0]["seed"] == 42 and sent[0]["temperature"] == 0
    assert model.cost_usd == pytest.approx(0.002)
    log = (tmp_path / "calls.jsonl").read_text()
    assert "test-key" not in log and "UnitProvider" in log
    assert model.metadata()["digest"] == "openrouter-slug:openai/gpt-oss-120b"
    assert "not_weight_hash" in model.metadata()["details"]["digest_kind"]


def test_budgets_missing_key_and_bad_output(dataset):
    data = payload(dataset)
    capped = OpenRouterModel(
        "openai/gpt-oss-120b", max_calls=1, api_key="test-key", transport=fake_transport([])
    )
    capped.generate("risk", data, AgentNote)
    with pytest.raises(RuntimeError, match="预算"):
        capped.generate("risk", data, AgentNote)
    costly = OpenRouterModel(
        "openai/gpt-oss-120b",
        max_calls=9,
        max_cost_usd=0.001,
        api_key="test-key",
        transport=fake_transport([], cost=0.002),
    )
    costly.generate("risk", data, AgentNote)
    with pytest.raises(RuntimeError, match="费用"):
        costly.generate("risk", data, AgentNote)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterModel("openai/gpt-oss-120b", api_key="").generate("risk", data, AgentNote)
    broken = OpenRouterModel(
        "openai/gpt-oss-120b", api_key="test-key", transport=fake_transport([], content="{}")
    )
    with pytest.raises(ValueError):
        broken.generate("risk", data, AgentNote)
    # 两次修复重试后仍不合规：共3次调用，全部记入审计，然后停止。
    assert [e["status"] for e in broken.audit] == ["invalid_output"] * 3
    assert broken.calls == 3
    with pytest.raises(ValueError):
        OpenRouterModel("no-provider-prefix")
    unknown = OpenRouterModel("openai/other", api_key="k", transport=fake_transport([]))
    with pytest.raises(ValueError, match="目录"):
        unknown.metadata()


def test_compare_cli_with_openrouter_backend(tmp_path, dataset, config, monkeypatch):
    from quant_research_lab import openrouter
    from quant_research_lab.cli import main

    sent: list[dict] = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter.OpenRouterModel.__init__,
        "__defaults__",
        (4, 42, 1.0, None, None, fake_transport(sent), "low", None, 2),
    )
    last = pd.Timestamp(dataset.dates[43])
    data = MarketDataset(dataset.frame[dataset.frame.trade_date <= last], dataset.manifest)
    data.frame.to_csv(tmp_path / "bars.csv", index=False)
    (tmp_path / "manifest.json").write_text(data.manifest.model_dump_json())
    (tmp_path / "config.json").write_text(
        config.model_copy(update={"max_model_calls": 20}).model_dump_json()
    )
    args = ["compare", "--backend", "openrouter", "--model", "openai/gpt-oss-120b"]
    args += ["--config", str(tmp_path / "config.json"), "--input", str(tmp_path / "bars.csv")]
    args += ["--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "pair")]
    args += ["--model-release", "2020-01-01", "--release-source", "https://example.org/unit"]
    assert main(args) == 0
    registration = json.loads((tmp_path / "pair" / "registration.json").read_text())
    assert registration["model"]["backend"] == "openrouter"
    assert len(sent) == 20
    assert "test-key" not in (tmp_path / "pair" / "model_calls.jsonl").read_text()
    init = ["forward-init", "--backend", "openrouter", "--model", "openai/gpt-oss-120b"]
    init += ["--model-release", "2020-01-01", "--release-source", "https://example.org/unit"]
    assert main([*init, "--output", str(tmp_path / "fwd")]) == 1


def test_strict_weights_conversion_and_provider_pinning(dataset):
    from quant_research_lab.openrouter import from_strict

    assert from_strict({"weights": [{"symbol": "A", "weight": 0.2}]}, True)["weights"] == {"A": 0.2}
    with pytest.raises(ValueError, match="多个权重"):
        from_strict({"weights": [{"symbol": "A", "weight": 0.1}] * 2}, True)
    sent: list[dict] = []
    model = OpenRouterModel(
        "openai/gpt-oss-120b",
        api_key="test-key",
        transport=fake_transport(sent),
        provider="openai/flex",
    )
    model.generate("risk", payload(dataset), AgentNote)
    assert sent[0]["provider"] == {
        "require_parameters": True,
        "only": ["openai/flex"],
        "allow_fallbacks": False,
    }


def test_invalid_output_is_repaired_once_and_logged(dataset):
    sent: list[dict] = []
    good = fake_transport(sent)
    replies = iter(['{"summary": "no role"}', None])

    def flaky(path, body, api_key):
        reply = next(replies)
        if reply is None:
            return good(path, body, api_key)
        sent.append(body)
        return {"choices": [{"message": {"content": reply}}], "usage": {"cost": 0.001}}

    model = OpenRouterModel("openai/gpt-oss-120b", max_calls=5, api_key="test-key", transport=flaky)
    note = model.generate("market", payload(dataset), AgentNote)
    assert note.role == "market"
    assert [e["status"] for e in model.audit] == ["invalid_output", "ok"]
    assert model.calls == 2 and model.cost_usd == pytest.approx(0.002)
    repair_messages = sent[1]["messages"]
    assert repair_messages[-2]["role"] == "assistant" and "schema" in repair_messages[-1]["content"]


def test_empty_reply_is_retried_with_changed_prompt(dataset):
    sent: list[dict] = []
    good = fake_transport(sent)
    replies = iter(["", None])

    def empty_first(path, body, api_key):
        reply = next(replies)
        if reply is None:
            return good(path, body, api_key)
        sent.append(body)
        return {"choices": [{"message": {"content": reply}}], "usage": {"cost": 0.001}}

    model = OpenRouterModel("openai/gpt-oss-120b", api_key="test-key", transport=empty_first)
    assert model.generate("risk", payload(dataset), AgentNote).role == "risk"
    assert [e["status"] for e in model.audit] == ["invalid_output", "ok"]
    assert sent[1]["messages"][-1]["content"].startswith("Your previous reply was empty")
    assert all(m["role"] != "assistant" for m in sent[1]["messages"])


def test_unsupported_sampling_parameters_are_not_sent(dataset):
    sent: list[dict] = []
    base = fake_transport(sent)

    def catalog_with_params(path, body, api_key):
        if body is None:
            return {
                "data": [
                    {**CATALOG["data"][0], "supported_parameters": ["seed", "structured_outputs"]}
                ]
            }
        return base(path, body, api_key)

    model = OpenRouterModel(
        "openai/gpt-oss-120b", api_key="test-key", transport=catalog_with_params
    )
    assert "temperature" not in model.metadata()["details"]["supported_parameters"]
    model.generate("risk", payload(dataset), AgentNote)
    assert "temperature" not in sent[0] and sent[0]["seed"] == 42
