"""Explicit optional SDK transport. Never changes the free default workflow."""

import contextlib
import importlib
import io
import json
from typing import Any
from unittest.mock import patch

from .tushare_source import SourceError, TushareSource


class TushareSDKSource(TushareSource):
    def _client(self) -> tuple[Any, Any]:
        try:
            ts = importlib.import_module("tushare")
        except ImportError:
            raise SourceError(
                "Install optional tushare SDK or use the existing a-share environment"
            ) from None
        try:
            pro = ts.pro_api(self.token)
            pro._DataApi__http_url = self.endpoint.rstrip("/")
            pro._DataApi__timeout = self.timeout
        except Exception:
            raise SourceError("SDK initialization failed; credentials suppressed") from None
        return ts, pro

    def _fetch(self, api: str, fields: str, params: dict[str, Any]) -> dict[str, Any]:
        _, pro = self._client()
        try:
            requests = importlib.import_module("requests")
            post = requests.post

            # SDK 1.4.29 posts to /api_name; reject redirects and non-2xx responses.
            def safe_post(*args: Any, **kwargs: Any) -> Any:
                kwargs["allow_redirects"] = False
                response = post(*args, **kwargs)
                if not 200 <= response.status_code < 300:
                    raise SourceError("SDK HTTP request rejected")
                return response

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                patch("tushare.pro.client.requests.post", side_effect=safe_post),
            ):
                frame = pro.query(api, fields=fields, **params)
            if frame is None:
                raise SourceError("SDK returned no table")
            table = json.loads(frame.to_json(orient="split", double_precision=15))
            return {"code": 0, "data": {"fields": table["columns"], "items": table["data"]}}
        except Exception:
            raise SourceError(
                "SDK query failed; response text and credentials suppressed"
            ) from None
