import json
from unittest.mock import MagicMock

import pytest

from quant_research_lab.agents import Committee, OllamaModel
from quant_research_lab.data import market_time
from quant_research_lab.models import AgentNote, Proposal
from quant_research_lab.risk import RiskState


class FakeModel:
    def __init__(self, unknown_citation=False, wrong_role=False, unsafe=False):
        self.unknown_citation = unknown_citation
        self.wrong_role = wrong_role
        self.unsafe = unsafe
        self.calls = []

    def generate(self, role, payload, schema):
        self.calls.append(role)
        ids = ("invented",) if self.unknown_citation else (payload["evidence"][0]["id"],)
        if schema is Proposal:
            return Proposal(
                weights={"SYN_A": 0.99 if self.unsafe else 0.1}, evidence_ids=ids, summary="test"
            )
        return AgentNote(
            role="market" if self.wrong_role else role, summary="test", evidence_ids=ids
        )


def test_four_model_roles_and_hard_veto(dataset, config):
    model = FakeModel(unsafe=True)
    result = Committee(config, model).decide(
        dataset.view(market_time(dataset.dates[-1])), 5, RiskState(dataset.symbols), 3
    )
    assert model.calls == ["market", "simulation", "risk", "manager"]
    assert result["approved_weights"] == {}
    assert any("cap" in item for item in result["risk_violations"])


@pytest.mark.parametrize("kwargs", [{"unknown_citation": True}, {"wrong_role": True}])
def test_model_protocol_failures(dataset, config, kwargs):
    with pytest.raises(ValueError):
        Committee(config, FakeModel(**kwargs)).decide(
            dataset.view(market_time(dataset.dates[-1])), 5, RiskState(dataset.symbols), 3
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://evil.example",
        "http://localhost/api",
        "http://user:pass@localhost",
        "http://localhost/?token=x",
    ],
)
def test_local_model_endpoint_boundary(url):
    with pytest.raises(ValueError):
        OllamaModel("demo", url)


def test_ollama_protocol_budget_and_schema(monkeypatch):
    note = AgentNote(role="market", summary="已核查", evidence_ids=("e1",))
    raw = json.dumps({"message": {"content": note.model_dump_json()}}).encode()
    response = MagicMock()
    response.__enter__.return_value.read.return_value = raw
    sent = []

    def fake_urlopen(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return response

    monkeypatch.setattr("quant_research_lab.agents.urlopen", fake_urlopen)
    model = OllamaModel("installed-local-model", max_calls=1)
    result = model.generate("market", {"evidence": []}, AgentNote)
    assert result == note
    assert sent[0][0]["stream"] is False
    assert sent[0][0]["format"]["type"] == "object"
    assert model.audit[0]["role"] == "market"
    with pytest.raises(RuntimeError):
        model.generate("risk", {}, AgentNote)
    response.__enter__.return_value.read.return_value = b"x" * 1_000_001
    with pytest.raises(ValueError):
        OllamaModel("model").generate("market", {}, AgentNote)


def test_missing_local_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(ValueError):
        OllamaModel()
