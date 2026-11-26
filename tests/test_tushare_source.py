import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_research_lab.data_bundle import load_quant_bundle, prepare_bundle
from quant_research_lab.tushare_source import SourceError, TushareSource, archive


def fake_network(monkeypatch, payload):
    response = MagicMock()
    response.__enter__.return_value = BytesIO(json.dumps(payload).encode())
    opener = MagicMock()
    opener.open.return_value = response
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: opener)
    return opener


def test_secret_free_cache_offline_reuse_and_tampering(tmp_path, monkeypatch):
    opener = fake_network(
        monkeypatch, {"code": 0, "data": {"fields": ["ts_code"], "items": [["000001.SZ"]]}}
    )
    source = TushareSource(tmp_path, token="secret-test-only", interval=0)
    expected = source.query("daily", ts_code="000001.SZ")
    request = opener.open.call_args.args[0]
    assert json.loads(request.data)["token"] == "secret-test-only"
    path = next(tmp_path.glob("*.json"))
    assert "secret-test-only" not in path.read_text()
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    pd.testing.assert_frame_equal(
        TushareSource(tmp_path).query("daily", ts_code="000001.SZ"), expected
    )
    assert opener.open.call_count == 1
    record = json.loads(path.read_text())
    record["data"]["items"] = [["tampered"]]
    path.write_text(json.dumps(record))
    with pytest.raises(SourceError, match="integrity"):
        source.query("daily", ts_code="000001.SZ")


def test_credentials_expiry_and_response_errors_not_persisted(tmp_path, monkeypatch):
    fake_network(monkeypatch, {"code": 0, "data": {"fields": ["token"], "items": [["secret"]]}})
    source = TushareSource(tmp_path, token="secret", interval=0)
    source.query("token_expire")
    assert not list(tmp_path.glob("*.json"))
    fake_network(monkeypatch, {"code": -1, "msg": "secret"})
    with pytest.raises(SourceError) as caught:
        source.query("daily")
    assert "secret" not in str(caught.value)
    with pytest.raises(ValueError):
        TushareSource(tmp_path, endpoint="http://example.com")
    with pytest.raises(ValueError):
        TushareSource(tmp_path, endpoint="https://user:password@example.com")
    with pytest.raises(ValueError):
        source.query("arbitrary_api")


def test_pagination_detects_repeated_page(tmp_path, monkeypatch):
    source = TushareSource(tmp_path, token="test")
    monkeypatch.setattr(source, "query", lambda *a, **k: pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(SourceError, match="repeated"):
        source.pages("daily", limit=2)


def raw_archive(directory):
    directory.mkdir()
    records = []
    tables = {
        "trade_cal": pd.DataFrame(
            {"cal_date": ["20240102", "20240103", "20240104", "20240105"], "is_open": [1, 1, 1, 1]}
        ),
        "daily": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": ["20240102", "20240103", "20240104"],
                "open": [10, 11, 12],
                "high": [10, 11, 12],
                "low": [10, 11, 12],
                "close": [10, 11, 12],
                "vol": [100, 200, 300],
            }
        ),
        "adj_factor": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": ["20240102", "20240103", "20240104"],
                "adj_factor": [2, 2, 2],
            }
        ),
        "stk_limit": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": ["20240102", "20240103", "20240104"],
                "up_limit": [11, 12.1, 13.2],
                "down_limit": [9, 9.9, 10.8],
            }
        ),
    }
    for api, frame in tables.items():
        path = directory / f"{api}.csv"
        frame.to_csv(path, index=False)
        records.append(
            {
                "api": api,
                "file": path.name,
                "rows": len(frame),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (directory / "archive_manifest.json").write_text(
        json.dumps({"status": "complete", "completed": records})
    )


def test_actual_two_project_contract_and_visibility(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "alpha-evidence-lab" / "src"))
    from alpha_evidence_lab.bundle import load_bundle

    raw_archive(tmp_path / "raw")
    metadata = prepare_bundle(tmp_path / "raw", tmp_path / "bundle")
    paths = load_bundle(tmp_path / "bundle")
    assert len(load_quant_bundle(tmp_path / "bundle").frame) == 3
    assert paths[0].bars[0].opening == 20
    assert paths[0].bars[0].up_limit == 22
    assert metadata["historical_universe_complete"] is False
    assert pd.read_csv(tmp_path / "bundle" / "bars.csv").volume.tolist() == [10000, 20000, 30000]
    prepare_bundle(tmp_path / "raw", tmp_path / "delayed", visibility="next_open")
    with pytest.raises(ValueError, match="requires close_assumed"):
        load_bundle(tmp_path / "delayed")
    csv = tmp_path / "bundle" / "alpha" / "000001.SZ.csv"
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(tmp_path / "bundle")


def test_archive_resumable_inventory(tmp_path, monkeypatch):
    source = TushareSource(tmp_path / "cache", token="test", interval=0)
    monkeypatch.setattr(source, "pages", lambda *a, **k: pd.DataFrame({"x": [1]}))
    result = archive(source, tmp_path / "raw", "20240101", "20240103", ["000001.SZ"], ["000300.SH"])
    assert result["status"] == "complete" and len(result["completed"]) == 10
    assert "test" not in (tmp_path / "raw" / "archive_manifest.json").read_text()
    with pytest.raises(ValueError, match="plan changed"):
        archive(source, tmp_path / "raw", "20240101", "20240104", ["000001.SZ"], ["000300.SH"])


def test_historical_union_downloads_former_members_not_only_current_symbols(tmp_path, monkeypatch):
    source = TushareSource(tmp_path / "cache", token="test", interval=0)
    requests = []

    def pages(api, **params):
        requests.append((api, params))
        if api == "index_weight":
            return pd.DataFrame({"con_code": ["600001.SH", "000001.SZ"]})
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(source, "pages", pages)
    archive(
        source,
        tmp_path / "raw",
        "20180101",
        "20180131",
        ["000001.SZ"],
        ["000300.SH"],
        all_index_members=True,
    )
    assert {p["ts_code"] for api, p in requests if api == "daily"} == {"600001.SH", "000001.SZ"}


def test_out_of_window_source_data_are_rejected(tmp_path, monkeypatch):
    fake_network(
        monkeypatch, {"code": 0, "data": {"fields": ["trade_date"], "items": [["20250101"]]}}
    )
    source = TushareSource(tmp_path, token="test", interval=0)
    with pytest.raises(SourceError, match="outside"):
        source.query("daily", start_date="20240101", end_date="20240131")
    assert not list(tmp_path.glob("*.json"))


def test_more_than_thirty_symbols_do_not_overflow_citation_schema(dataset, config):
    from quant_research_lab.agents import Committee
    from quant_research_lab.data import MarketDataset, market_time
    from quant_research_lab.risk import RiskState

    base = dataset.frame[dataset.frame.ts_code == "SYN_A"]
    frame = pd.concat([base.assign(ts_code=f"TEST_{i:03d}") for i in range(40)], ignore_index=True)
    wide = MarketDataset(frame, dataset.manifest)
    decision = Committee(config).decide(
        wide.view(market_time(wide.dates[60])), 5, RiskState(wide.symbols), 1
    )
    assert len(decision["evidence"]) == 40
    assert all(len(note["evidence_ids"]) <= 30 for note in decision["notes"])
    ids = decision["manager"]["evidence_ids"]
    for symbol in decision["manager"]["weights"]:
        assert any(item.startswith(f"price:{symbol}:") for item in ids)
