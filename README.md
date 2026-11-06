# FPL projections

Expected FPL points per player for the next 16 gameweeks (the site shows the next 8 by default; the wildcard planner uses all 16), rebuilt from scratch.

- Data: `olbauday/FPL-Core-Insights` (refreshes several times a day, more on match days). Everything is keyed on `player_code` / team `code`; ids are not stable across seasons.
- Model: team attack/defence model (xG, shots on target, defensive actions) -> player shares with recency decay -> minutes/availability -> exact point distributions under the official scoring (`config/scoring_snapshot.json`, refreshed from the live API when reachable).
- Output: `scripts/run_production.py` writes AES-GCM-encrypted JSON (`FPL_SITE_PASSWORD` secret); `site/index.html` decrypts it in the browser.
- Automation: `.github/workflows/update.yml` runs every hour and deploys to GitHub Pages. No paid services, no AI calls at runtime.
- Backtest: `python scripts/run_backtest.py <data_repo> --quick`. New or low-history players can be corrected in `config/overrides.json`.

## Squad builder
`site/index.html` picks the best 15 in the browser: 2 GKP / 5 DEF / 5 MID / 3 FWD, budget, max 3 per club, best legal XI each gameweek plus a doubled captain, bench weighted 0.1. Players can be kept or swapped out. `tests/exact_squad.py` solves the same objective exactly with HiGHS to check the heuristic (matched exactly at 3 and 8 gameweeks, within 0.2% at 1).

## Teams and transfers
`config/teams.json` lists the FPL team ids the site offers (add a friend with one line: `{"id": 123456, "label": "Name"}`). Each run reads those squads from the public FPL endpoints (last finished gameweek's picks plus any transfers already made, selling prices from purchase history, bank, estimated free transfers). The Squad tab then shows the best 0-3 transfers and what each is worth: projected gain over the horizon minus 4 points per transfer beyond the free ones. `tests/exact_squad.py --team` checks the planner against an exact MILP (matched in every case tested). Free transfers are an estimate and can be corrected in the page.

## Penalties and bonus points

- `fplmodel/penalties.py`: penalties are modelled apart from open-play goals. A club's penalty rate comes from its own
  decayed history; the taker share comes from penalties taken *for that club*. A transfer
  resets the evidence (the player is flagged as a mover). `scripts/eval_penalties.py` scores the method on real penalties
  (86 events: actual taker picked first 69% of the time vs 28% for xG shares).
- `fplmodel/bonus.py`: bonus points come from a per-match simulation of BPS (weights fitted by `scripts/fit_bps.py`
  into `config/bps_model.json`). Backtest: calibrated overall and slightly better MAE than the old regression, similar RMSE
  (bonus is noisy). The old regression remains as a fallback if `bps_model.json` is missing.

## Minutes

Each match a player is in one of three states: starts, comes off the bench, or is unused. A walk-forward
multinomial model (`fplmodel/minutes.py`) gives the three probabilities from his recent starts, appearances and
tenure. Starts come from FPL's own `starts` count per gameweek (2025/26 on; earlier rows fall back to "played
60+"). Minutes are then conditional: a starter reaches 60 with probability `q60s` and plays `m60` minutes, an early
hook plays `m_hook`, a substitute plays `m_sub` (all decayed per-player averages shrunk to the position).
Expected minutes = P(start)·minutes-if-start + P(sub)·minutes-if-sub, which also feeds the 60+ and short-appearance
points. Team totals are blended three-quarters of the way towards a normal match (`NORM_STRENGTH = 0.75`).

The start / sub / unused model is a small gradient-boosted classifier (`MINUTES_MODEL=lr` switches to logistic
regression) fed with: his own recent starts and appearances at three memory lengths; **club rest** (days since and
until the club's next match in any competition, from the full calendar including European and cup fixtures, and
matches in the last 14 days); **competition for the place** (starting rates of available teammates in his position);
and **return from absence** (share of recent matches unavailable, days since he last played, appearances since a
21+ day break, length of his last appearance). It is evaluated walk-forward, and the per-fixture probabilities are
computed for each player in each target fixture (`Context._minutes_fx`). Backtest (15 gameweeks, 12.5k player-weeks):
minutes MAE 12.5 → 11.4, RMSE 20.9 → 20.3, bias about 0, points MAE 0.989 → 0.969. Most of the gain comes from the
return-from-absence features, then competition for places and rest. A rating-gap (Elo) feature added nothing.

## Price alerts

`.github/workflows/prices.yml` runs every 30 minutes, reads the official FPL price-change predictor figures from the public bootstrap-static feed
(`scripts/fetch_prices.py`) and force-pushes one file, `prices.json`, to the orphan branch `prices`. The site fetches it from
raw.githubusercontent.com on load, so prices stay fresh without a redeploy. It flags suggested signings about to rise, players you would sell
(or own) about to fall, and players whose price has already moved since the last full update.

## Live scores

The **Live** tab shows the gameweek in play: scores, goals, assists and cards per match, the top scorers, and live points for each team in `config/teams.json` (provisional bonus from BPS until the official bonus lands, and automatic substitutions once a team's games are over). `.github/workflows/live.yml` runs every five minutes, fetches the official Fantasy feed and publishes `live.json` to the `live` branch during a match window only; the site reads it from there and re-reads it every minute while the tab is open. GitHub's scheduler can run a few minutes late, so expect it to trail the official site slightly.

### Faster live scores (Cloudflare Worker)

`worker/` is a small Cloudflare Worker that sits between the page and the official FPL API. Browsers cannot read that API directly (no CORS permission header), so the page asks the Worker; the Worker asks FPL, keeps each answer at Cloudflare's edge for 20 to 600 seconds depending on the endpoint, and returns it with the permission header added. It allows only GET, only six read-only paths, and only requests from the site's own address. The page then works out live points, bonus, auto-subs and league ranks itself (the JavaScript in `site/index.html` between the `LIVE-CALC` markers is a port of `scripts/fetch_live.py`; `tests/test_live_port.py` checks they agree) and refreshes every 30 seconds. If the Worker is not set or is down, it falls back to the five-minute GitHub file.

To switch it on: add repository secrets `CLOUDFLARE_API_TOKEN` (the "Edit Cloudflare Workers" token template) and `CLOUDFLARE_ACCOUNT_ID`, run **Deploy proxy Worker**, then put the Worker's address in `FPL_PROXY` in `site/index.html`.

