import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_research_lab.tushare_sdk import TushareSDKSource
from quant_research_lab.tushare_source import SourceError


def test_sdk_uses_explicit_client_and_does_not_persist_token(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=MagicMock()))
    pro = MagicMock()
    pro.query.return_value = pd.DataFrame({"ts_code": ["000001.SZ"]})
    factory = MagicMock(return_value=pro)
    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=factory))
    source = TushareSDKSource(
        tmp_path,
        token="private-test-only",
        endpoint="http://example.test:24145",
        allow_http=True,
        interval=0,
    )
    with pytest.MonkeyPatch.context() as m:
        # No actual SDK package or HTTP request is necessary in this unit test.
        m.setattr(
            "quant_research_lab.tushare_sdk.patch",
            lambda *a, **k: __import__("contextlib").nullcontext(),
        )
        frame = source.query("index_basic", limit=5)
    assert len(frame) == 1
    factory.assert_called_once_with("private-test-only")
    assert pro._DataApi__http_url == "http://example.test:24145"
    pro.query.assert_called_once_with("index_basic", fields="", limit=5)
    record = next(tmp_path.glob("*.json")).read_text()
    assert "private-test-only" not in record
    assert json.loads(record)["request"]["transport"] == "TushareSDKSource"


def test_sdk_errors_do_not_expose_response_text(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=MagicMock()))
    pro = MagicMock()
    pro.query.side_effect = RuntimeError("credential-in-response")
    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=lambda token: pro))
    monkeypatch.setattr(
        "quant_research_lab.tushare_sdk.patch",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )
    with pytest.raises(SourceError) as caught:
        TushareSDKSource(tmp_path, token="private", interval=0).query("daily")
    assert "credential-in-response" not in str(caught.value)
    assert not list(tmp_path.glob("*.json"))
