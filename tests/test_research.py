import json
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_research_lab.analytics import bootstrap_scenarios, performance, target_weights
from quant_research_lab.cli import main
from quant_research_lab.data import MarketDataset, market_time
from quant_research_lab.reporting import render_html, save_report
from quant_research_lab.research import run_research, select_window
from quant_research_lab.store import AuditStore


def test_metrics_hand_example():
    stats = performance([100, 110, 99])
    assert stats["total_return"] == pytest.approx(-0.01)
    assert stats["max_drawdown"] == pytest.approx(0.1)
    assert performance([100, 100, 100])["sharpe"] is None
    for bad in [[100], [100, 0], [100, np.nan]]:
        with pytest.raises(ValueError):
            performance(bad)
    assert target_weights({"A": -0.1}, 2, 0.3, 0.6) == {}
    assert target_weights({"B": 0.2, "A": 0.2}, 1, 0.3, 0.6) == {"A": 0.3}


def test_bootstrap_fixed_seed_and_limits():
    first = bootstrap_scenarios([0.01, -0.02] * 10, 7, 30, 5)
    assert first == bootstrap_scenarios([0.01, -0.02] * 10, 7, 30, 5)
    assert first["p05"] <= first["median"] <= first["p95"]
    with pytest.raises(ValueError):
        bootstrap_scenarios([np.nan] * 20, 7, 30, 5)


def test_oos_cannot_change_training_selection(dataset, config):
    training = dataset.dates[: config.training_sessions]
    original = select_window(dataset, training, config, 0)
    frame = dataset.frame.copy()
    future = frame.trade_date > pd.Timestamp(training[-1])
    frame.loc[future, ["open", "high", "low", "close"]] *= 10
    changed = MarketDataset(frame, dataset.manifest)
    assert original == select_window(changed, training, config, 0)
    first_cutoff = market_time(training[-1])
    pd.testing.assert_frame_equal(
        dataset.view(first_cutoff).panel(), changed.view(first_cutoff).panel()
    )


def test_rolls_trials_and_disclosures(result, config):
    assert result["parameter_trials"] == len(result["folds"]) * len(config.windows)
    assert result["backtest_backend"] == "deterministic_rules_not_llm"
    for fold in result["folds"]:
        assert fold["train_end"] < fold["test_start"] <= fold["test_end"]
    assert all(row["cash_fraction"] >= -1e-8 for row in result["curve"])
    assert result["manifest"]["historical_universe_complete"] is False
    assert result["forecast_scores"]
    for memory in result["forecast_scores"]:
        original = next(item for item in result["forecasts"] if item["id"] == memory["id"])
        assert pd.Timestamp(memory["available_at"]) >= pd.Timestamp(original["forecast"]["due_at"])


def test_store_atomicity_and_asof(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite")
    now = market_time(pd.Timestamp("2024-01-02").date())
    store.record_session("run", "first", now, {"ok": True}, [{"id": "m1", "value": 1}])
    assert store.memories_as_of("run", now - timedelta(seconds=1)) == []
    assert len(store.memories_as_of("run", now)) == 1
    with pytest.raises(sqlite3.IntegrityError):
        store.record_session("run", "second", now, {"ok": True}, [{"id": "m1"}])
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    with pytest.raises(ValueError):
        store.memories_as_of("run", now.replace(tzinfo=None))
    store.save_forecast("run", "f1", now, {"prediction": 0.1})


def test_report_artifacts_and_reproducibility(result, dataset, tmp_path):
    copy = {**result, "limitations": ["<script>alert('bad')</script>"]}
    save_report(copy, dataset.frame, tmp_path)
    report = (tmp_path / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert '<svg role="img"' in report
    assert 'src="http' not in report
    assert (tmp_path / "curve.csv").exists()
    record = json.loads((tmp_path / "reproducibility.json").read_text())
    assert record["run_id"] == result["run_id"]
    assert "research.json" in record["artifacts_sha256"]
    loaded = MarketDataset.read(tmp_path / "input_bars.csv", tmp_path / "input_manifest.json")
    assert loaded.digest == dataset.digest
    assert "人工行情" in render_html(result)


def test_cli_input_and_output_protection(tmp_path, dataset, config):
    source = tmp_path / "bars.csv"
    manifest = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    dataset.frame.to_csv(source, index=False)
    manifest.write_text(dataset.manifest.model_dump_json())
    config_path.write_text(config.model_dump_json())
    output = tmp_path / "run"
    args = [
        "research",
        "--input",
        str(source),
        "--manifest",
        str(manifest),
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert main(args) == 1
    assert main(["research", "--output", str(tmp_path / "bad")]) == 1
    assert (output / "audit.sqlite").exists()


def test_insufficient_data(dataset):
    from quant_research_lab.models import ResearchConfig

    with pytest.raises(ValueError):
        run_research(dataset, ResearchConfig(training_sessions=120))
