# FPL projections

Expected FPL points per player for the next 8 gameweeks, rebuilt from scratch.

- Data: `olbauday/FPL-Core-Insights` (refreshes 07:30 and 17:30 UTC). Everything is keyed on `player_code` / team `code`; ids are not stable across seasons.
- Model: team attack/defence model (xG, shots on target, defensive actions) -> player shares with recency decay -> minutes/availability -> exact point distributions under the official scoring (`config/scoring_snapshot.json`, refreshed from the live API when reachable).
- Output: `scripts/run_production.py` writes AES-GCM-encrypted JSON (`FPL_SITE_PASSWORD` secret); `site/index.html` decrypts it in the browser.
- Automation: `.github/workflows/update.yml` runs twice a day and deploys to GitHub Pages. No paid services, no AI calls at runtime.
- Backtest: `python scripts/run_backtest.py <data_repo> --quick`. New or low-history players can be corrected in `config/overrides.json`.
