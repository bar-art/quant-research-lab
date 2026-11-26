# Training data indexing — 2026-11-25

The training hot loop now uses immutable per-symbol array snapshots and per-date bar snapshots, rather than constructing and copying the full visible panel for every candidate/date. Availability is filtered before selecting lagged observations, including non-monotone publication delays. Treat MarketDataset.frame as read-only; construct a new dataset after edits.

A fixed local benchmark on 1,260 synthetic sessions, four symbols, one 80-session training fold, and the default candidate windows measured 0.680 seconds before and 0.229 seconds after. Selected window, all trial metrics and bootstrap scenarios were identical. This measures the training selector, not a full five-year study or a 300-stock production workload. The general AsOfView and LLM replay still have separate overheads. Tests compare indexed scores to the original panel implementation under delayed availability, stale symbols and multiple training starts.

Remaining: whole-study scaling benchmarks, LLM interruption/resume, daily historical-universe eligibility including removal/liquidation rules, and conservative publication-delay sensitivity analysis. The baseline commits preserve the original behavior for comparison.

## Whole-study benchmark — 2026-11-26

Workload: HS300 historical universe from the ashare-data-pipeline Tushare snapshot, 668 symbols (including later removals and delistings), 1,737,796 daily rows, 2015-01-05 to 2026-11-24, 141 rolling test folds, 423 candidate evaluations, rule backend.

| Version | Wall time | Output |
|---|---|---|
| Before | 78 min 32 s | run `1eb9853014240be85521` |
| After | 8 min 18 s | identical run id; every `research.json` field equal |

What changed, in profile order:

1. `AsOfView` no longer copies the visible panel for each cutoff. Views created by `MarketDataset.view()` read per-symbol visible row positions (`visible_rows`) and per-symbol frames; `momentum`, `history`, `universe`, `evidence` and `signal_age` use them. A view built directly from a DataFrame keeps the original row-filter implementation, and `tests/test_view_index.py` asserts the two agree under non-monotone publication delays, stale symbols, universe changes and unknown symbols.
2. `MarketDataset.scores()` slices the date-sorted frame to the requested date range and computes all symbols in one vectorised pass instead of a Python loop over symbols. The existing `scores == features(panel)` reference tests still pass.
3. The research loop asked SQLite for all matured memories twice per decision but used only the count and the last 20. It now uses `COUNT(*)` and `ORDER BY … DESC LIMIT 20`; a test checks both against the full query.

On a 2-year slice of the same data the time went 348 s → 77 s.
