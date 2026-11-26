"""Validate archived Tushare tables and export one hash-checked bundle for both labs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import MarketDataset
from .models import DataManifest


def prepare_bundle(
    archive: Path, output: Path, *, visibility: str = "close_assumed"
) -> dict[str, Any]:
    if visibility not in {"close_assumed", "next_open"}:
        raise ValueError("Unknown visibility policy")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new; previous research is never overwritten")
    meta = json.loads((archive / "archive_manifest.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("Incomplete archive; resume acquisition first")
    tables: dict[str, list[pd.DataFrame]] = {}
    for item in meta["completed"]:
        path = (archive / item["file"]).resolve()
        if path.parent != archive.resolve():
            raise ValueError("Unsafe archive path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Raw archive hash mismatch")
        if item["rows"]:
            frame = pd.read_csv(path, dtype=str)
            if len(frame) != item["rows"]:
                raise ValueError("Archive row count mismatch")
            tables.setdefault(item["api"], []).append(frame)
    if not tables.get("daily") or not tables.get("trade_cal"):
        raise ValueError("Missing daily prices or trading calendar")
    daily = pd.concat(tables["daily"], ignore_index=True)
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol"}
    if required - set(daily) or daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Missing price columns or duplicate price keys")
    calendar = pd.concat(tables["trade_cal"], ignore_index=True)
    dates = sorted(calendar.loc[calendar.is_open == "1", "cal_date"].unique())
    if not dates:
        raise ValueError("No trading sessions")
    for _, group in daily.groupby("ts_code"):
        expected = {d for d in dates if group.trade_date.min() <= d <= group.trade_date.max()}
        if expected - set(group.trade_date):
            raise ValueError("Missing sessions: explicit suspended-day valuation data required")
    limits = (
        pd.concat(tables.get("stk_limit", []), ignore_index=True)
        if tables.get("stk_limit")
        else pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    )
    factors = (
        pd.concat(tables.get("adj_factor", []), ignore_index=True)
        if tables.get("adj_factor")
        else pd.DataFrame()
    )
    if factors.empty:
        raise ValueError(
            "Adjustment factors missing; raw ex-right jumps cannot be silently treated as returns"
        )
    data = daily.merge(
        limits[["ts_code", "trade_date", "up_limit", "down_limit"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    data = data.merge(
        factors[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    factor = pd.to_numeric(data.adj_factor, errors="raise")
    if factor.isna().any() or (factor <= 0).any() or not np.isfinite(factor).all():
        raise ValueError("Unknown or invalid adjustment factors")
    for column in ["open", "high", "low", "close", "up_limit", "down_limit"]:
        data[column] = pd.to_numeric(data[column], errors="raise") * factor
    data["volume"] = pd.to_numeric(data.vol, errors="raise") * 100  # Tushare vol: hands.
    data["is_suspended"] = False  # Every retained row is a source daily trading record.
    if tables.get("suspend_d"):
        suspensions = pd.concat(tables["suspend_d"], ignore_index=True)
        if {"suspend_type", "ts_code", "trade_date"} <= set(suspensions):
            # A daily suspension record may be intraday. Conservatively block the whole day.
            blocked = set(
                map(
                    tuple,
                    suspensions.loc[
                        suspensions.suspend_type == "S", ["ts_code", "trade_date"]
                    ].to_numpy(),
                )
            )
            data["is_suspended"] = [
                pair in blocked for pair in zip(data.ts_code, data.trade_date, strict=True)
            ]
    data["price_limit_exempt"] = False  # Unknown limits fail closed in the execution engine.
    next_day = dict(zip(dates[:-1], dates[1:], strict=True))
    if visibility == "next_open":
        if not set(data.trade_date) <= set(next_day):
            raise ValueError("next_open requires a calendar extending beyond the last price date")
        data["available_at"] = data.trade_date.map(next_day) + "T09:00:00+08:00"
    else:
        data["available_at"] = data.trade_date + "T15:00:00+08:00"
    data["trade_date"] = pd.to_datetime(data.trade_date, format="%Y%m%d")
    manifest = DataManifest(
        kind="user_supplied",
        source="Tushare-compatible archive; adjusted research prices; observed daily rows only",
    )
    dataset = MarketDataset(data, manifest)
    return write_bundle(
        dataset,
        output,
        visibility=visibility,
        source_digest=hashlib.sha256((archive / "archive_manifest.json").read_bytes()).hexdigest(),
    )


def write_bundle(
    dataset: MarketDataset,
    output: Path,
    *,
    visibility: str = "close_assumed",
    source_digest: str = "",
) -> dict[str, Any]:
    """Source-neutral writer used by verified exports; never fetches network data."""
    if visibility not in {"close_assumed", "next_open"}:
        raise ValueError("Unknown visibility policy")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new")
    for code in dataset.symbols:
        if (
            Path(code).name != code
            or code in {".", ".."}
            or "\\" in code
            or any(ord(c) < 32 for c in code)
        ):
            raise ValueError("Unsafe symbol filename")
    manifest = dataset.manifest
    output.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(output / "bars.csv", index=False)
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    files = []
    alpha = output / "alpha"
    alpha.mkdir()
    for symbol, group in dataset.frame.groupby("ts_code"):
        path = alpha / f"{symbol}.csv"
        frame = group.rename(columns={"trade_date": "date"})
        frame.to_csv(
            path,
            index=False,
            columns=[
                "date",
                "open",
                "close",
                "available_at",
                "is_suspended",
                "up_limit",
                "down_limit",
                "price_limit_exempt",
            ],
        )
        files.append(
            {
                "file": path.relative_to(output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = {
        "schema": "research_bundle_v1",
        "source": manifest.source,
        "visibility": visibility,
        "rows": len(dataset.frame),
        "unknown_limit_rows": int(dataset.frame.up_limit.isna().sum()),
        "suspended_rows": int(dataset.frame.is_suspended.sum()),
        "symbols": list(dataset.symbols),
        "alpha_files": files,
        "quant_files": [
            {"file": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
            for name in ["bars.csv", "manifest.json"]
        ],
        "historical_universe_complete": False,
        "corporate_actions_complete": False,
        "archive_manifest_sha256": source_digest,
        "limitations": [
            "Monthly constituent snapshots are not daily eligibility",
            "Selected symbols do not remove survivorship bias",
            "Adjusted prices and fractional execution are research approximations",
            "Original publication vintages and total-return corporate actions are not certified",
            "Alpha currently accepts close_assumed only; next_open is a quant sensitivity bundle",
        ],
    }
    (output / "bundle.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def load_quant_bundle(directory: Path) -> MarketDataset:
    root = directory.resolve()
    metadata = json.loads((root / "bundle.json").read_text())
    if metadata.get("schema") != "research_bundle_v1":
        raise ValueError("Unsupported research bundle version")
    inventory = metadata.get("quant_files", [])
    if sorted(item["file"] for item in inventory) != ["bars.csv", "manifest.json"]:
        raise ValueError("Bundle needs exactly bars.csv and manifest.json hashes")
    for item in inventory:
        path = (root / item["file"]).resolve()
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Bundle artifact hash mismatch")
    return MarketDataset.read(root / "bars.csv", root / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--visibility", choices=["close_assumed", "next_open"], default="close_assumed"
    )
    args = parser.parse_args()
    try:
        result = prepare_bundle(args.archive, args.output, visibility=args.visibility)
        print(f"Prepared {result['rows']} rows; historical_universe_complete=False")
        return 0
    except (OSError, ValueError, KeyError):
        print("Bundle validation failed; check archive coverage, columns and hashes")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
