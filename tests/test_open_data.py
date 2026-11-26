import json
from pathlib import Path

import pytest

from quant_research_lab.data_bundle import load_quant_bundle, write_bundle
from quant_research_lab.open_data import run_pipeline


def workspace(tmp_path, ashare="a-share-quant-agent"):
    for name in [ashare, "quant-research-lab", "alpha-evidence-lab"]:
        python = tmp_path / name / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.touch()
    return tmp_path


def exporter(monkeypatch, dataset):
    calls = []

    def child(command, cwd, log, **kwargs):
        calls.append(command)
        log.write_text("offline test fixture; not a live request")
        if "--target" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            dataset.frame.to_csv(output / "bars.csv", index=False)
            (output / "manifest.json").write_text(dataset.manifest.model_dump_json())
            (output / "export_report.json").write_text(json.dumps({"fixture": True}))

    monkeypatch.setattr("quant_research_lab.open_data.run_child", child)
    return calls


def test_cache_only_pipeline_keeps_contract_without_token(tmp_path, monkeypatch, dataset):
    root = workspace(tmp_path / "workspace")
    calls = exporter(monkeypatch, dataset)
    output = tmp_path / "result"
    result = run_pipeline(
        root, output, list(dataset.symbols), "2020-01-02", "2024-12-31", export_only=True
    )
    assert result["mode"] == "verified_cache_only"
    assert len(calls) == 2  # Cache verification followed by explicit BaoStock export.
    assert "-c" in calls[0] and "baostock" in calls[1]
    assert all("tushare" not in " ".join(command).lower() for command in calls)
    assert load_quant_bundle(output / "bundle").digest == dataset.digest
    with pytest.raises(ValueError, match="new output"):
        run_pipeline(
            root, output, list(dataset.symbols), "2020-01-02", "2024-12-31", export_only=True
        )


def test_short_paths_are_not_silently_dropped(tmp_path, monkeypatch, dataset):
    root = workspace(tmp_path / "workspace")
    exporter(monkeypatch, dataset)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="630"):
        run_pipeline(root, output, list(dataset.symbols), "2020-01-02", "2024-12-31")
    assert json.loads((output / "pipeline.json").read_text())["status"] == "exported"


def test_bundle_writer_does_not_overwrite_existing_data(tmp_path, dataset):
    output = tmp_path / "bundle"
    write_bundle(dataset, output)
    with pytest.raises(ValueError, match="new"):
        write_bundle(dataset, output)


@pytest.mark.parametrize("folder", ["ashare-data-pipeline", "a-share-quant-agent"])
def test_data_repository_found_under_new_or_old_folder_name(tmp_path, monkeypatch, dataset, folder):
    root = workspace(tmp_path / "workspace", ashare=folder)
    calls = exporter(monkeypatch, dataset)
    run_pipeline(
        root, tmp_path / "out", list(dataset.symbols), "2020-01-02", "2024-12-31", export_only=True
    )
    assert calls[0][0] == str(root.resolve() / folder / ".venv" / "bin" / "python")
