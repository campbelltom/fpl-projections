# FPL projections

Expected FPL points per player for the next 8 gameweeks, rebuilt from scratch.

- Data: `olbauday/FPL-Core-Insights` (refreshes 07:30 and 17:30 UTC). Everything is keyed on `player_code` / team `code`; ids are not stable across seasons.
- Model: team attack/defence model (xG, shots on target, defensive actions) -> player shares with recency decay -> minutes/availability -> exact point distributions under the official scoring (`config/scoring_snapshot.json`, refreshed from the live API when reachable).
- Output: `scripts/run_production.py` writes AES-GCM-encrypted JSON (`FPL_SITE_PASSWORD` secret); `site/index.html` decrypts it in the browser.
- Automation: `.github/workflows/update.yml` runs twice a day and deploys to GitHub Pages. No paid services, no AI calls at runtime.
- Backtest: `python scripts/run_backtest.py <data_repo> --quick`. New or low-history players can be corrected in `config/overrides.json`.

## Squad builder
`site/index.html` picks the best 15 in the browser: 2 GKP / 5 DEF / 5 MID / 3 FWD, budget, max 3 per club, best legal XI each gameweek plus a doubled captain, bench weighted 0.1. Players can be kept or swapped out. `tests/exact_squad.py` solves the same objective exactly with HiGHS to check the heuristic (matched exactly at 3 and 8 gameweeks, within 0.2% at 1).

## Teams and transfers
`config/teams.json` lists the FPL team ids the site offers (add a friend with one line: `{"id": 123456, "label": "Name"}`). Each run reads those squads from the public FPL endpoints (last finished gameweek's picks plus any transfers already made, selling prices from purchase history, bank, estimated free transfers). The Squad tab then shows the best 0-3 transfers and what each is worth: projected gain over the horizon minus 4 points per transfer beyond the free ones. `tests/exact_squad.py --team` checks the planner against an exact MILP (matched in every case tested). Free transfers are an estimate and can be corrected in the page.

## Penalties and bonus points

- `fplmodel/penalties.py`: penalties are modelled apart from open-play goals. A club's penalty rate comes from its own
  decayed history; the taker share comes from penalties taken *for that club* plus FPL's `penalties_order`. A transfer
  resets the evidence (the player is flagged as a mover). `scripts/eval_penalties.py` scores the method on real penalties
  (86 events: actual taker picked first 69% of the time vs 28% for xG shares).
- `fplmodel/bonus.py`: bonus points come from a per-match simulation of BPS (weights fitted by `scripts/fit_bps.py`
  into `config/bps_model.json`). Backtest: calibrated overall and slightly better MAE than the old regression, similar RMSE
  (bonus is noisy). The old regression remains as a fallback if `bps_model.json` is missing.
