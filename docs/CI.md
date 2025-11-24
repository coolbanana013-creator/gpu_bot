CI & Testing Notes
==================

This repo contains a CI workflow to run tests and a debug wrapper.

Running tests offline (recommended for CI):

```powershell
$env:SKIP_LOAD_MARKETS='1'
$env:DATA_FETCHER_SKIP_LOAD_MARKETS='1'
$env:PYTHONIOENCODING='utf-8'
python -m pytest -q
```

This prevents CCXT from calling `load_markets()` in offline environments and avoids Unicode console encoding issues on Windows.

To prefer perpetual/swap markets when using `DataFetcher`, set the env variable `DATA_FETCHER_CONTRACT_TYPE`:

```powershell
$env:DATA_FETCHER_CONTRACT_TYPE='perpetual'
```

For CI, there's an additional on-demand integration job (manual trigger) which runs tests that require live exchange access `RUN_INTEGRATION_TESTS=1`.
