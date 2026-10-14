"""Production run: data -> predictions for the next 8 gameweeks -> (encrypted) JSON for the site.

    python scripts/run_production.py <data_dir> --out site/data [--no-encrypt]

No network is needed except optional live FPL status/news (falls back to the latest data snapshot)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fplmodel.config import CURRENT_SEASON, HORIZON_GWS, Params  # noqa: E402
from fplmodel.data import current_gw  # noqa: E402
from fplmodel.engine import Context  # noqa: E402
from fplmodel.export import encrypt_json  # noqa: E402
from fplmodel.predict import BonusModel, aggregate_gameweek  # noqa: E402
from fplmodel.scoring import load_rules  # noqa: E402

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"


def live_bootstrap() -> dict | None:
    import urllib.request
    try:
        req = urllib.request.Request(BOOTSTRAP, headers={"User-Agent": "fpl-projections/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001 - offline / blocked is a normal fallback
        print(f"live FPL API unavailable ({e}); using data snapshot", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    ap.add_argument("--no-encrypt", action="store_true")
    a = ap.parse_args()
    notes: list[str] = []

    ctx = Context.build(a.src)
    s = ctx.seasons[CURRENT_SEASON]
    gw0 = current_gw(s.matches)
    if gw0 is None:
        raise SystemExit("no unplayed gameweek found - season over?")
    gws = [g for g in range(gw0, gw0 + HORIZON_GWS) if g <= 38]
    first_ko = s.matches.loc[s.matches["gameweek"] == gw0, "kickoff_time"].min()
    # use the moment of the run (or the GW start, if it already began) as the information cutoff
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(first_ko.tz) if first_ko.tzinfo else pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    asof = min(now, first_ko) if (s.matches[s.matches["gameweek"] == gw0]["finished"].sum() == 0) else now

    # roster, status and news: live API if reachable, otherwise last data snapshot (post-GW row gw0-1)
    players = s.players.set_index("player_code")
    team_now = players["team_code"].astype(int)
    st = ctx.status[(ctx.status["season"] == CURRENT_SEASON) & (ctx.status["gw"] == gw0 - 1)][["player_code", "status", "chance", "news"]]
    price = pd.Series(dtype=float)
    bs = live_bootstrap()
    if bs:
        el = pd.DataFrame(bs["elements"])
        tcode = {t["id"]: t["code"] for t in bs["teams"]}
        st = pd.DataFrame({"player_code": el["code"], "status": el["status"],
                           "chance": el["chance_of_playing_next_round"], "news": el["news"]})
        tn = el.set_index("code")["team"].map(tcode)
        team_now = tn.where(tn.notna(), team_now.reindex(tn.index)).dropna().astype(int)
        price = (el.set_index("code")["now_cost"] / 10.0)
        rules = load_rules(live=True)
    else:
        notes.append("Live FPL API not reachable on this run: status/news from the latest data snapshot.")
        rules = load_rules(live=False)

    overrides = {}
    f = ROOT / "config" / "overrides.json"
    if f.exists():
        overrides = {k: v for k, v in json.loads(f.read_text()).items() if not k.startswith("_")}

    params, bonus = Params.load(), BonusModel.load()
    d = ctx.predict(asof, CURRENT_SEASON, gws, params, rules, st, bonus, team_now=team_now, overrides=overrides)
    g = aggregate_gameweek(d)

    # ---- assemble output
    teams = s.teams.set_index("code")
    hist = ctx.opp[ctx.opp["minutes"] > 0].groupby("player_code").size()
    fx = {}
    for (pc, gw), r in d.groupby(["player_code", "gw"]):
        fx[(pc, gw)] = [f"{teams.loc[int(o), 'short_name']}({'H' if h else 'A'})" for o, h in zip(r["opp"], r["is_home"])]
    stx = st.drop_duplicates("player_code").set_index("player_code")
    out_players = []
    for pc, grp in g.groupby("player_code"):
        grp = grp.set_index("gw").reindex(gws)
        pl = players.loc[pc]
        row = {
            "code": int(pc), "name": pl["web_name"], "pos": pl["position"], "team": teams.loc[int(team_now[pc]), "short_name"],
            "price": float(price.get(pc, np.nan)) if len(price) else None,
            "status": stx["status"].get(pc, "a"), "news": str(stx["news"].get(pc) if pd.notna(stx["news"].get(pc)) else "")[:120],
            "ep": [round(float(x), 2) if pd.notna(x) else 0.0 for x in grp["ep"]],
            "p_play": [round(float(x), 2) if pd.notna(x) else 0.0 for x in grp["p_play"]],
            "fx": [", ".join(fx.get((pc, gw), [])) or "BLANK" for gw in gws],
            "comp": {k[4:]: round(float(grp["pts_" + k[4:]].iloc[0]), 2) for k in
                     ["pts_appearance", "pts_goals", "pts_assists", "pts_clean_sheet", "pts_goals_conceded", "pts_saves",
                      "pts_defensive", "pts_cards", "pts_bonus"] if "pts_" + k[4:] in grp and pd.notna(grp["pts_" + k[4:]].iloc[0])},
            "sd": round(float(grp["ep_sd"].iloc[0]), 2) if pd.notna(grp["ep_sd"].iloc[0]) else None,
            "low_history": bool(hist.get(pc, 0) < 5),
        }
        row["total"] = round(float(sum(row["ep"])), 2)
        if row["total"] >= 0.05:
            out_players.append(row)
    out_players.sort(key=lambda r: -r["total"])
    n_low = sum(r["low_history"] for r in out_players)
    if n_low:
        notes.append(f"{n_low} players have fewer than 5 league appearances on record (flagged low_history).")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT_SEASON, "current_gw": int(gw0), "horizon": gws, "model_version": params.version,
        "gw_dates": {int(gw): str(s.matches.loc[s.matches["gameweek"] == gw, "kickoff_time"].min()) for gw in gws},
        "notes": notes, "players": out_players,
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.no_encrypt:
        (out / "predictions.json").write_text(json.dumps(payload, separators=(",", ":")))
    else:
        pw = os.environ.get("FPL_SITE_PASSWORD")
        if not pw:
            raise SystemExit("FPL_SITE_PASSWORD not set (use --no-encrypt for a local preview)")
        (out / "predictions.enc.json").write_text(json.dumps(encrypt_json(payload, pw)))
    print(f"GW{gw0}..{gws[-1]}: {len(out_players)} players; top: " +
          ", ".join(f"{r['name']} {r['total']}" for r in out_players[:5]))


if __name__ == "__main__":
    main()
