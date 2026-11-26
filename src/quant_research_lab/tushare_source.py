"""Tushare-compatible, read-only acquisition with secret-free resumable raw cache."""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

ALLOWED = {
    "token_expire",
    "index_basic",
    "stock_basic",
    "trade_cal",
    "index_weight",
    "daily",
    "adj_factor",
    "stk_limit",
    "suspend_d",
    "daily_basic",
    "fina_indicator",
}


class SourceError(RuntimeError):
    """Sanitized error: never echo server messages, request bodies or credentials."""


class TushareSource:
    def __init__(
        self,
        cache: Path,
        *,
        token: str | None = None,
        endpoint: str | None = None,
        allow_http: bool = False,
        timeout: float = 20,
        interval: float = 0.3,
    ) -> None:
        self.token = token or os.environ.get("TUSHARE_TOKEN", "")
        self.endpoint = endpoint or os.environ.get("TUSHARE_BASE_URL", "https://api.tushare.pro")
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Invalid data endpoint; credentials and query strings are forbidden")
        if url.scheme == "http" and not allow_http:
            raise ValueError("HTTP requires explicit --allow-http authorization")
        if timeout <= 0 or interval < 0:
            raise ValueError("Invalid timeout or request interval")
        self.cache, self.timeout, self.interval = cache, timeout, interval
        self._last = 0.0

    def query(self, api: str, *, fields: str = "", **params: Any) -> pd.DataFrame:
        if api not in ALLOWED:
            raise ValueError("API not on the read-only allowlist")
        if {"token", "password", "secret"} & {key.lower() for key in params}:
            raise ValueError("Credentials must not appear in query parameters")
        identity = {
            "api": api,
            "fields": fields,
            "params": params,
            "endpoint_sha256": hashlib.sha256(self.endpoint.encode()).hexdigest(),
            "transport": type(self).__name__,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if api != "token_expire" and path.exists():
            record = json.loads(path.read_text())
            raw = record["data"]
            if (
                record["request"] != identity
                or record["sha256"]
                != hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            ):
                raise SourceError("Cached data failed integrity validation")
            frame = pd.DataFrame(raw["items"], columns=raw["fields"])
            self._validate_dates(frame, params)
            return frame
        if not self.token:
            raise SourceError(
                "Set TUSHARE_TOKEN; existing market-data cache remains usable offline"
            )
        time.sleep(max(0, self.interval - (time.monotonic() - self._last)))
        self._last = time.monotonic()
        result = self._fetch(api, fields, params)
        if result.get("code") != 0 or not isinstance(result.get("data"), dict):
            raise SourceError("Data server rejected the query; response text suppressed")
        raw = result["data"]
        if not isinstance(raw.get("fields"), list) or not isinstance(raw.get("items"), list):
            raise SourceError("Malformed tabular response")
        if api != "token_expire" and self.token and self.token in json.dumps(raw):
            raise SourceError("Unexpected credential in market response; not cached")
        frame = pd.DataFrame(raw["items"], columns=raw["fields"])
        self._validate_dates(frame, params)
        if api != "token_expire":
            self.cache.mkdir(parents=True, exist_ok=True)
            record = {
                "request": identity,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "data": raw,
                "sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False))
            temporary.replace(path)
        return frame

    def _fetch(self, api: str, fields: str, params: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            {"api_name": api, "token": self.token, "params": params, "fields": fields}
        ).encode()
        request = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            # No redirect: the access token must not be forwarded to another endpoint.
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(
                    self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any
                ) -> None:
                    return None

            with urllib.request.build_opener(NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                result = json.load(response)
        except (OSError, ValueError, urllib.error.URLError):
            raise SourceError(
                "Data server connection/JSON failure; no credentials logged"
            ) from None
        return dict(result)

    @staticmethod
    def _validate_dates(frame: pd.DataFrame, params: dict[str, Any]) -> None:
        for column in ["trade_date", "cal_date"]:
            if column not in frame or frame.empty:
                continue
            dates = pd.to_datetime(frame[column], format="%Y%m%d", errors="raise")
            if dates.isna().any():
                raise SourceError("Missing source date")
            if (
                params.get("start_date") and (dates < pd.Timestamp(params["start_date"])).any()
            ) or (params.get("end_date") and (dates > pd.Timestamp(params["end_date"])).any()):
                raise SourceError("Server returned dates outside requested window")

    def pages(self, api: str, *, limit: int = 3000, **params: Any) -> pd.DataFrame:
        pages = []
        seen: set[str] = set()
        for offset in range(0, 3_000_000, limit):
            frame = self.query(api, limit=limit, offset=offset, **params)
            if frame.empty:
                break
            fingerprint = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
            if fingerprint in seen:
                raise SourceError("Server repeated a page; refusing silently truncated history")
            seen.add(fingerprint)
            pages.append(frame)
            if len(frame) < limit:
                break
        else:
            raise SourceError("Pagination safety limit reached")
        return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def archive(
    source: TushareSource,
    output: Path,
    start: str,
    end: str,
    symbols: list[str],
    indices: list[str],
    *,
    all_index_members: bool = False,
    include_stock_basic: bool = True,
) -> dict[str, Any]:
    """Save raw data first; monthly weights are snapshots, not daily effective membership."""
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start > end")
    output.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, dict[str, Any]]] = [
        (
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": start,
                "end_date": (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y%m%d"),
            },
        ),
        *[
            ("stock_basic", {"list_status": status})
            for status in (["L", "D", "P"] if include_stock_basic else [])
        ],
    ]
    for index in indices:
        for month in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
            tasks.append(
                (
                    "index_weight",
                    {
                        "index_code": index,
                        "start_date": month.start_time.strftime("%Y%m%d"),
                        "end_date": min(month.end_time, pd.Timestamp(end)).strftime("%Y%m%d"),
                    },
                )
            )
    metadata_count = len(tasks)

    def append_prices(selected: list[str]) -> None:
        for symbol in selected:
            for api in ["daily", "adj_factor", "stk_limit", "suspend_d", "daily_basic"]:
                tasks.append((api, {"ts_code": symbol, "start_date": start, "end_date": end}))

    if not all_index_members:
        append_prices(symbols)
    plan = {
        "version": 1,
        "start": start,
        "end": end,
        "symbols": symbols,
        "indices": indices,
        "all_index_members": all_index_members,
        "include_stock_basic": include_stock_basic,
        "endpoint_sha256": hashlib.sha256(source.endpoint.encode()).hexdigest(),
    }
    plan_path = output / "archive_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Archive plan changed; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty archive has no acquisition plan; use a new directory")
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    records: list[dict[str, Any]] = []
    for api, params in tasks:
        frame = source.pages(api, **params)
        name = (
            api
            + "_"
            + hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
            + ".csv"
        )
        frame.to_csv(output / name, index=False)
        records.append(
            {
                "api": api,
                "params": params,
                "file": name,
                "rows": len(frame),
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            }
        )
        (output / "archive_manifest.json").write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "start": start,
                    "end": end,
                    "completed": records,
                    "planned_queries": len(tasks),
                    "historical_universe_complete": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if all_index_members and len(records) == metadata_count:
            members = set(symbols)
            for item in records:
                if item["api"] == "index_weight" and item["rows"]:
                    weights = pd.read_csv(output / item["file"], dtype=str)
                    if "con_code" not in weights:
                        raise SourceError("Historical weights lack con_code; cannot build union")
                    members.update(weights.con_code)
            if not any(item["api"] == "index_weight" and item["rows"] for item in records):
                raise SourceError(
                    "No historical index constituents; refusing today's universe fallback"
                )
            append_prices(sorted(members))
        print(f"Archived {len(records)}/{len(tasks)} {api}: {len(frame)} rows", flush=True)
    result = {
        "status": "complete",
        "start": start,
        "end": end,
        "completed": records,
        "planned_queries": len(tasks),
        "historical_universe_complete": False,
        "note": (
            "Monthly weights do not establish daily/original-vintage membership; "
            "selected-symbol prices are not a full-market archive."
        ),
    }
    (output / "archive_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only data archive; token only from environment"
    )
    parser.add_argument("command", choices=["probe", "archive"])
    parser.add_argument("--cache", type=Path, default=Path("output/tushare_cache"))
    parser.add_argument("--output", type=Path, default=Path("output/tushare_archive"))
    parser.add_argument("--start", default="20150101")
    parser.add_argument("--end", default="20261124")
    parser.add_argument("--symbols", nargs="+", default=["000001.SZ", "600000.SH"])
    parser.add_argument("--indices", nargs="+", default=["000300.SH", "000905.SH", "000016.SH"])
    parser.add_argument("--allow-http", action="store_true")
    parser.add_argument("--transport", choices=["http", "sdk"], default="http")
    parser.add_argument("--request-timeout", type=float, default=20)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Calendar and selected-symbol market data; omit stock metadata and index weights",
    )
    parser.add_argument(
        "--all-index-members",
        action="store_true",
        help="Archive prices for union of all retrieved historical constituent snapshots",
    )
    args = parser.parse_args()
    try:
        source: TushareSource
        if args.transport == "sdk":
            from .tushare_sdk import TushareSDKSource

            source = TushareSDKSource(
                args.cache, allow_http=args.allow_http, timeout=args.request_timeout
            )
        else:
            source = TushareSource(
                args.cache, allow_http=args.allow_http, timeout=args.request_timeout
            )
        if args.command == "probe":
            frame = source.query("token_expire")
            print(
                json.dumps(
                    {"api": "token_expire", "rows": len(frame), "columns": list(frame.columns)}
                )
            )
            frame = source.query("index_basic", limit=5)
            print(
                json.dumps(
                    {"api": "index_basic", "rows": len(frame), "columns": list(frame.columns)}
                )
            )
        else:
            if args.core_only and args.all_index_members:
                raise ValueError("core-only and all-index-members are incompatible")
            archive(
                source,
                args.output,
                args.start,
                args.end,
                args.symbols,
                [] if args.core_only else args.indices,
                all_index_members=args.all_index_members,
                include_stock_basic=not args.core_only,
            )
        return 0
    except (SourceError, ValueError):
        print(
            "Acquisition stopped. Check connectivity, expiry and permissions. "
            "Completed cache retained; server messages and token suppressed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
