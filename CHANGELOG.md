# Changelog

## v1.8.3 - 2025-11-24
- Add CI wrapper for running debug/inspect/parity scripts (`scripts/tools/run_all_debug_tests.py`) and a GitHub Actions workflow `ci-clean.yml` that runs tests offline and provides an on-demand integration test.
- DataFetcher: improved `markets_by_id` mapping and alias support, added env var `DATA_FETCHER_CONTRACT_TYPE` to prefer `perpetual`, `future`, or `spot` markets.
- DataFetcher: support `SKIP_LOAD_MARKETS` and `DATA_FETCHER_SKIP_LOAD_MARKETS` env var for favoring offline test runs.
- Fix parity for ADX in `indicator_calculator.py` by implementing a kernel-like CPU ADX to match GPU float32 & Wilder smoothing semantics.
- Add tests for DataFetcher mapping, debug wrapper, and parity tests.
- Minor cleanup: convert script duplicates to stubs and created `deprecated/scripts_backup` for previous versions.
