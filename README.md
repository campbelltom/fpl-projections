# FPL projections

Expected FPL points per player for the next 8 gameweeks, rebuilt from scratch.

- Data: `olbauday/FPL-Core-Insights` (refreshes 07:30 and 17:30 UTC). Everything is keyed on `player_code` / team `code`; ids are not stable across seasons.
- Model: team attack/defence model (xG, shots on target, defensive actions) -> player shares with recency decay -> minutes/availability -> exact point distributions under the official scoring (`config/scoring_snapshot.json`, refreshed from the live API when reachable).
- Output: `scripts/run_production.py` writes AES-GCM-encrypted JSON (`FPL_SITE_PASSWORD` secret); `site/index.html` decrypts it in the browser.
- Automation: `.github/workflows/update.yml` runs twice a day and deploys to GitHub Pages. No paid services, no AI calls at runtime.
- Backtest: `python scripts/run_backtest.py <data_repo> --quick`. New or low-history players can be corrected in `config/overrides.json`.

## Squad builder
`site/index.html` picks the best 15 in the browser: 2 GKP / 5 DEF / 5 MID / 3 FWD, budget, max 3 per club, best legal XI each gameweek plus a doubled captain, bench weighted 0.1. Players can be kept or swapped out. `tests/exact_squad.py` solves the same objective exactly with HiGHS to check the heuristic (matched exactly at 3 and 8 gameweeks, within 0.2% at 1).
