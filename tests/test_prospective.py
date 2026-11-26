import json
from datetime import timedelta

import pytest
from test_llm_replay import CashModel, provenance

from quant_research_lab.data import MarketDataset, market_time
from quant_research_lab.models import DataManifest
from quant_research_lab.prospective import (
    create_commitment,
    load_commitment,
    prospective_step,
    settle_prospective,
)


def setup_forward(tmp_path, dataset, config):
    data = MarketDataset(
        dataset.frame,
        DataManifest(kind="user_supplied", source="unit-test fixture; not real market evidence"),
    )
    plan = config.model_copy(update={"max_model_calls": 8})
    first, second, third = dataset.dates[70:73]
    calendar = [
        {
            "decision_at": market_time(a).isoformat(),
            "execution_at": market_time(b, 9).replace(minute=30).isoformat(),
            "mark_at": market_time(b).isoformat(),
        }
        for a, b in [(first, second), (second, third)]
    ]
    registered = market_time(first) - timedelta(hours=1)
    create_commitment(tmp_path, plan, provenance(), dataset.symbols, calendar, now=registered)
    return data, plan, calendar, registered


def test_true_forward_lifecycle_without_future_grading(tmp_path, dataset, config):
    data, plan, calendar, registered = setup_forward(tmp_path / "study", dataset, config)
    root = tmp_path / "study"
    from quant_research_lab.prospective import aware

    with pytest.raises(ValueError, match="实时决策窗口"):
        prospective_step(root, data, CashModel(), now=registered)
    now = aware(calendar[0]["decision_at"]) + timedelta(seconds=1)
    result = prospective_step(root, data, CashModel(), now=now)
    assert result["llm"]["approved_weights"] == {}
    with pytest.raises(ValueError):
        prospective_step(root, data, CashModel(), now=now)
    with pytest.raises(ValueError, match="尚未到期"):
        settle_prospective(root, data, now=now)
    marked = aware(calendar[0]["mark_at"]) + timedelta(seconds=1)
    grade = settle_prospective(root, data, now=marked)
    assert grade["status"] == "in_progress"
    prospective_step(root, data, CashModel(), now=marked)
    final = settle_prospective(root, data, now=aware(calendar[1]["mark_at"]) + timedelta(seconds=1))
    assert final["status"] == "complete" and final["settled_sessions"] == 2
    assert final["increment"]["status"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="不重复评分"):
        settle_prospective(root, data, now=aware(calendar[1]["mark_at"]) + timedelta(days=1))


def test_future_registration_and_tamper(tmp_path, dataset, config):
    data, plan, calendar, now = setup_forward(tmp_path / "first", dataset, config)
    root = tmp_path / "first"
    with pytest.raises(ValueError):
        create_commitment(
            tmp_path / "late",
            plan,
            provenance(),
            dataset.symbols,
            calendar,
            now=now + timedelta(days=2),
        )
    body = json.loads((root / "commitment.json").read_text())
    body["universe"] = ["CHANGED"]
    (root / "commitment.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="被修改"):
        load_commitment(root)


def test_no_synthetic_no_late_decision_and_no_silent_retry(tmp_path, dataset, config):
    from quant_research_lab.prospective import aware

    data, plan, calendar, now = setup_forward(tmp_path / "study", dataset, config)
    root = tmp_path / "study"
    moment = aware(calendar[0]["decision_at"]) + timedelta(seconds=1)
    with pytest.raises(ValueError, match="人工行情"):
        prospective_step(root, dataset, CashModel(), now=moment)
    with pytest.raises(ValueError, match="实时决策窗口"):
        prospective_step(
            root, data, CashModel(), now=aware(calendar[0]["execution_at"]) + timedelta(seconds=1)
        )
    with pytest.raises(ValueError):
        prospective_step(root, data, CashModel(fail_at=1), now=moment)
    with pytest.raises(ValueError, match="缺少按时完成"):
        settle_prospective(root, data, now=aware(calendar[0]["mark_at"]) + timedelta(seconds=1))


def test_code_lock_and_missing_prices(tmp_path, dataset, config, monkeypatch):
    from quant_research_lab.prospective import aware

    data, plan, calendar, now = setup_forward(tmp_path / "study", dataset, config)
    root = tmp_path / "study"
    original = load_commitment(root)
    monkeypatch.setattr("quant_research_lab.prospective.code_hash", lambda: "new-code")
    with pytest.raises(ValueError, match="代码发生变化"):
        load_commitment(root)
    monkeypatch.setattr("quant_research_lab.prospective.code_hash", lambda: original["code_hash"])
    prospective_step(
        root, data, CashModel(), now=aware(calendar[0]["decision_at"]) + timedelta(seconds=1)
    )
    partial = MarketDataset(data.frame[data.frame.ts_code != dataset.symbols[0]], data.manifest)
    with pytest.raises(ValueError, match="行情缺失"):
        settle_prospective(root, partial, now=aware(calendar[0]["mark_at"]) + timedelta(seconds=1))
