"""Free A-share workflow using the existing BaoStock adapter and verified cache."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import MarketDataset
from .data_bundle import write_bundle

ASHARE_DIRS = ("ashare-data-pipeline", "a-share-quant-agent")
DEFAULT_CODES = ["000001.SZ", "300750.SZ", "600000.SH", "600518.SH", "600519.SH", "688981.SH"]
CACHE_CHECK = """
import json,sys
from pathlib import Path
from datetime import date
from quant_agent.data.cache import LocalCache
root=Path(sys.argv[1])
if not (root/"catalog.duckdb").is_file():
    raise ValueError("Cache database missing; run with --online to acquire data")
cache=LocalCache(root)
start,end=date.fromisoformat(sys.argv[2]),date.fromisoformat(sys.argv[3])
for code in json.loads(sys.argv[4]):
    if cache.gaps("baostock",code,start,end):
        raise ValueError("Cache coverage missing for "+code+"; use --online explicitly")
    if cache.read("baostock","bars",code) is None:
        raise ValueError("Cached bars missing")
calendar=cache.read("baostock","calendar","")
instruments=cache.read("baostock","instruments","")
if calendar is None or instruments is None:
    raise ValueError("Cached calendar or instrument metadata missing")
if calendar.trade_date.min().date()>start or calendar.trade_date.max().date()<end:
    raise ValueError("Calendar does not cover requested interval")
print("Verified BaoStock cache coverage and file integrity")
"""


def run_child(
    command: list[str],
    cwd: Path,
    log: Path,
    *,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> None:
    with log.open("w") as handle:
        result = subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, env=env
        )
    if result.returncode:
        raise ValueError(f"Step failed; inspect {log}")


def run_pipeline(
    workspace: Path,
    output: Path,
    codes: list[str],
    start: str,
    end: str,
    *,
    online: bool = False,
    export_only: bool = False,
) -> dict[str, Any]:
    import os

    workspace, output = workspace.resolve(), output.resolve()
    # 数据仓库已更名为 ashare-data-pipeline；本地仍叫旧名 a-share-quant-agent 的目录也可用。
    ashare = next(
        (workspace / name for name in ASHARE_DIRS if (workspace / name).is_dir()),
        workspace / ASHARE_DIRS[0],
    )
    quant, alpha = workspace / "quant-research-lab", workspace / "alpha-evidence-lab"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    for root in [ashare, quant, alpha]:
        if not (root / ".venv/bin/python").is_file():
            raise ValueError(f"Install the project's Python environment first: {root.name}")
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "DATA_DIR": str(ashare / "data")}
    python = str(ashare / ".venv/bin/python")
    if not online:
        run_child(
            [python, "-c", CACHE_CHECK, str(ashare / "data"), start, end, json.dumps(codes)],
            ashare,
            output / "cache_check.log",
            env=environment,
        )
    exported = output / "provider_export"
    run_child(
        [
            python,
            "-m",
            "quant_agent.cli",
            "export",
            "--provider",
            "baostock",
            "--start",
            start,
            "--end",
            end,
            "--codes",
            *codes,
            "--target",
            "quant-research-lab",
            "--output",
            str(exported),
        ],
        ashare,
        output / "export.log",
        env=environment,
    )
    dataset = MarketDataset.read(exported / "bars.csv", exported / "manifest.json")
    if set(dataset.symbols) != set(codes):
        raise ValueError("Export silently omitted requested symbols")
    bundle = output / "bundle"
    report = exported / "export_report.json"
    write_bundle(dataset, bundle, source_digest=hashlib.sha256(report.read_bytes()).hexdigest())
    result: dict[str, Any] = {
        "provider": "baostock",
        "mode": "online_gaps_allowed" if online else "verified_cache_only",
        "start": start,
        "end": end,
        "rows": len(dataset.frame),
        "symbols": list(dataset.symbols),
        "dataset_sha256": dataset.digest,
        "historical_universe_complete": False,
        "corporate_actions_complete": False,
        "status": "exported",
        "limits": [
            "Selected historical sample, not a survivorship-free universe",
            "Daily close availability is assumed; fractional shares/proportional costs",
            "Alpha and quant use different research windows and are not competing strategies",
        ],
    }
    status = output / "pipeline.json"
    status.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if not export_only:
        # Do not silently discard short/delisted paths to satisfy the evidence gate.
        if dataset.frame.groupby("ts_code").size().min() < 630:
            raise ValueError(
                "Alpha needs 630 rows/path; use export-only or preregister another plan"
            )
        run_child(
            [
                sys.executable,
                "-m",
                "quant_research_lab.cli",
                "research",
                "--bundle",
                str(bundle),
                "--output",
                str(output / "quant"),
            ],
            quant,
            output / "quant.log",
        )
        run_child(
            [
                str(alpha / ".venv/bin/python"),
                "-m",
                "alpha_evidence_lab.cli",
                "research",
                "--bundle",
                str(bundle),
                "--output",
                str(output / "alpha"),
            ],
            alpha,
            output / "alpha.log",
        )
        run_child(
            [
                str(alpha / ".venv/bin/python"),
                "-m",
                "alpha_evidence_lab.cli",
                "verify",
                "--output",
                str(output / "alpha"),
            ],
            alpha,
            output / "verify.log",
        )
        result["status"] = "both_studies_completed_and_alpha_verified"
        status.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Free BaoStock pipeline; no token or Tushare")
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / ("baostock_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--codes", nargs="+", default=DEFAULT_CODES)
    parser.add_argument("--start", default="2020-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--online", action="store_true", help="Explicitly allow fetching missing BaoStock data"
    )
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.workspace,
            args.output,
            args.codes,
            args.start,
            args.end,
            online=args.online,
            export_only=args.export_only,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Stopped: {type(error).__name__}: {str(error)[:240]}")
        print("Artifacts retained. No synthetic substitution. Use a NEW output directory.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
