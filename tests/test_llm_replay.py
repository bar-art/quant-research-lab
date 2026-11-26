import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_research_lab.agents import OllamaModel
from quant_research_lab.comparison import compare_research, paired_increment, planned_model_calls
from quant_research_lab.data import MarketDataset
from quant_research_lab.models import AgentNote, Proposal
from quant_research_lab.research import run_research


class CashModel:
    max_calls = 1000

    def __init__(self, unsafe=False, fail_at=None):
        self.calls = 0
        self.inputs = []
        self.unsafe, self.fail_at = unsafe, fail_at

    def generate(self, role, payload, schema):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("controlled model failure")
        self.inputs.append((role, payload))
        ids = (payload["evidence"][0]["id"],)
        if schema is Proposal:
            return Proposal(
                weights={"SYN_A": 0.99} if self.unsafe else {},
                evidence_ids=ids,
                summary="independent cash decision",
            )
        return AgentNote(role=role, evidence_ids=ids, summary="known data only")


def short_data(dataset):
    last = pd.Timestamp(dataset.dates[43])
    return MarketDataset(dataset.frame[dataset.frame.trade_date <= last], dataset.manifest)


def provenance():
    return {
        "name": "test-double",
        "digest": "unit-test-only",
        "released_at": "2020-01-01T00:00:00+00:00",
        "release_source": "unit-test-fixture",
    }


def test_model_drives_every_replay_decision_and_trade(dataset, config):
    data = short_data(dataset)
    model = CashModel()
    llm = run_research(data, config, model=model, model_provenance=provenance())
    rules = run_research(data, config)
    assert llm["backtest_backend"] == "llm_committee_replay"
    assert model.calls == planned_model_calls(data, config) == 20
    assert llm["trades"] == [] and llm["metrics"]["total_return"] == 0
    assert rules["trades"]
    assert llm["run_id"] != rules["run_id"]
    assert {item["as_of"] for item in llm["decision_log"]} >= {
        item["as_of"] for item in llm["sessions"]
    }
    for _role, payload in model.inputs:
        cutoff = pd.Timestamp(payload["as_of"])
        assert all(pd.Timestamp(item["available_at"]) <= cutoff for item in payload["evidence"])
        assert pd.Timestamp(payload["training_context"]["train_end"]).date() <= cutoff.date()
    assert llm["forecast_backend"] == "rule_momentum_diagnostic_not_llm_forecast"


def test_unsafe_model_cannot_disable_risk(dataset, config):
    data = short_data(dataset)
    result = run_research(data, config, model=CashModel(unsafe=True))
    assert all(item["approved_weights"] == {} for item in result["decision_log"])
    assert all("position_cap:SYN_A" in item["risk_violations"] for item in result["decision_log"])


def test_pair_artifacts_and_budget_preflight(tmp_path, dataset, config):
    data = short_data(dataset)
    model = CashModel()
    model.max_calls = 1
    with pytest.raises(ValueError, match="预算"):
        compare_research(data, config, model, provenance(), tmp_path / "too-small")
    assert model.calls == 0
    model.max_calls = 20
    result = compare_research(data, config, model, provenance(), tmp_path / "pair")
    assert result["training_windows_equal"] is True
    assert result["increment"]["status"] == "insufficient_evidence"
    assert result["registration"]["evidence_class"] == "synthetic_engineering_only"
    assert (tmp_path / "pair/llm/report.html").exists()
    assert "逐时点大模型" in (tmp_path / "pair/comparison.html").read_text()
    assert "规则信号诊断" in (tmp_path / "pair/llm/report.html").read_text()


def test_model_failure_is_not_silent_rules_fallback(tmp_path, dataset, config):
    data = short_data(dataset)
    model = CashModel(fail_at=5)
    with pytest.raises(ValueError):
        compare_research(data, config, model, provenance(), tmp_path / "failed")
    assert (tmp_path / "failed/rules/research.json").exists()
    assert not (tmp_path / "failed/comparison.json").exists()
    assert json.loads((tmp_path / "failed/failed.json").read_text())["completed_pair"] is False


def test_release_guard_and_paired_interval(tmp_path, dataset, config):
    meta = {**provenance(), "released_at": "2025-01-01T00:00:00+00:00"}
    with pytest.raises(ValueError, match="早于模型发布"):
        compare_research(
            short_data(dataset), config, CashModel(), meta, tmp_path / "before-release"
        )
    model = [1.01**i for i in range(61)]
    rules = [1.005**i for i in range(61)]
    result = paired_increment(model, rules, 42, samples=100)
    assert result["daily_mean_interval"][0] > 0
    assert result == paired_increment(model, rules, 42, samples=100)


def test_real_client_metadata_digest_and_error_journal(monkeypatch, tmp_path):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"models": [{"name": "local", "digest": "new"}]}
    ).encode()
    monkeypatch.setattr("quant_research_lab.agents.urlopen", lambda *args, **kwargs: response)
    model = OllamaModel("local", expected_digest="old", audit_path=tmp_path / "events.jsonl")
    assert model.metadata()["digest"] == "new"
    with pytest.raises(ValueError, match="发生变化"):
        model.generate("market", {}, AgentNote)
    assert model.audit[-1]["status"] == "model_changed"
    response.__enter__.return_value.read.return_value = b'{"message":{"content":"not-json"}}'
    failing = OllamaModel("local", audit_path=tmp_path / "fail.jsonl")
    with pytest.raises(ValueError):
        failing.generate("market", {}, AgentNote)
    assert json.loads((tmp_path / "fail.jsonl").read_text())["status"] == "error"


def test_manager_request_constrains_symbols_size_and_notes(monkeypatch):
    response = MagicMock()
    proposal = Proposal(weights={"SYN_A": 0.2}, evidence_ids=("e1",), summary="valid")
    response.__enter__.return_value.read.return_value = json.dumps(
        {"message": {"content": proposal.model_dump_json()}}
    ).encode()
    sent = []

    def fake_open(request, **kwargs):
        sent.append(json.loads(request.data))
        return response

    monkeypatch.setattr("quant_research_lab.agents.urlopen", fake_open)
    model = OllamaModel("unit-local")
    model.generate(
        "manager",
        {
            "momentum": {"SYN_A": 0.01, "SYN_B": 0.02},
            "policy": {"max_position": 0.3, "max_gross": 0.6},
            "top_k": 2,
            "evidence": [{"id": "e1"}],
        },
        Proposal,
    )
    weights = sent[0]["format"]["properties"]["weights"]
    assert weights["additionalProperties"] is False
    assert set(weights["properties"]) == {"SYN_A", "SYN_B"}
    assert weights["properties"]["SYN_A"]["maximum"] == 0.3
    assert weights["maxProperties"] == 2
    assert sent[0]["options"]["num_predict"] == 384


def test_compare_cli_and_forward_dispatch(tmp_path, dataset, config, monkeypatch):
    from quant_research_lab.cli import main

    class LocalFake(CashModel):
        def __init__(self, **kwargs):
            super().__init__()
            self.max_calls = kwargs["max_calls"]
            self.expected_digest = None

        def metadata(self):
            return {"name": "unit-local", "digest": "unit-only-digest"}

    monkeypatch.setattr("quant_research_lab.cli.OllamaModel", LocalFake)
    data = short_data(dataset)
    data.frame.to_csv(tmp_path / "bars.csv", index=False)
    (tmp_path / "manifest.json").write_text(data.manifest.model_dump_json())
    cfg = config.model_copy(update={"max_model_calls": 20})
    (tmp_path / "config.json").write_text(cfg.model_dump_json())
    common = [
        "--config",
        str(tmp_path / "config.json"),
        "--model-release",
        "2020-01-01",
        "--release-source",
        "https://example.org/unit-test",
    ]
    assert (
        main(
            [
                "compare",
                *common,
                "--input",
                str(tmp_path / "bars.csv"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--output",
                str(tmp_path / "pair"),
            ]
        )
        == 0
    )
    calendar = [
        {
            "decision_at": "2099-01-05T15:00:00+08:00",
            "execution_at": "2099-01-06T09:30:00+08:00",
            "mark_at": "2099-01-06T15:00:00+08:00",
        }
    ]
    (tmp_path / "calendar.json").write_text(json.dumps(calendar))
    root = tmp_path / "forward"
    assert (
        main(
            [
                "forward-init",
                *common,
                "--universe",
                *dataset.symbols,
                "--calendar",
                str(tmp_path / "calendar.json"),
                "--output",
                str(root),
            ]
        )
        == 0
    )
    for command in ["forward-step", "forward-settle"]:
        assert (
            main(
                [
                    command,
                    "--input",
                    str(tmp_path / "bars.csv"),
                    "--manifest",
                    str(tmp_path / "manifest.json"),
                    "--output",
                    str(root),
                ]
            )
            == 1
        )
    assert not (root / "forward_grade.json").exists()


def test_model_position_count_has_independent_runtime_guard(dataset, config):
    from quant_research_lab.agents import Committee
    from quant_research_lab.data import market_time
    from quant_research_lab.risk import RiskState

    class WideModel(CashModel):
        def generate(self, role, payload, schema):
            if schema is Proposal:
                return Proposal(
                    weights=dict.fromkeys(dataset.symbols, 0.1),
                    evidence_ids=(payload["evidence"][0]["id"],),
                    summary="too many positions",
                )
            return super().generate(role, payload, schema)

    result = Committee(config, WideModel()).decide(
        dataset.view(market_time(dataset.dates[-1])), 5, RiskState(dataset.symbols), 3
    )
    assert result["approved_weights"] == {} and "top_k_cap" in result["risk_violations"]


def test_cached_responses_resume_without_repeating_completed_calls(dataset, config, tmp_path):
    data = short_data(dataset)
    cache = tmp_path / "shared-cache"
    first = CashModel(fail_at=4)
    with pytest.raises(ValueError):
        compare_research(
            data, config, first, provenance(), tmp_path / "attempt1", model_cache=cache
        )
    resumed = CashModel()
    result = compare_research(
        data, config, resumed, provenance(), tmp_path / "attempt2", model_cache=cache
    )
    assert result["model_cache_hits"] == 3
    assert resumed.calls == result["actual_model_calls"] == 17
    assert (tmp_path / "attempt1" / "failed.json").exists()
    assert result["llm_metrics"]["total_return"] == 0
