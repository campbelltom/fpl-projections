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
from fplmodel.config import BUDGET, CURRENT_SEASON, HORIZON_GWS, SQUAD, Params  # noqa: E402
from fplmodel.data import current_gw  # noqa: E402
from fplmodel.penalties import team_summary
from fplmodel.engine import Context  # noqa: E402
from fplmodel.export import encrypt_json
from fplmodel.myteam import build_team  # noqa: E402
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
    st = ctx.status[(ctx.status["season"] == CURRENT_SEASON) & (ctx.status["gw"] == gw0 - 1)][["player_code", "status", "chance", "news", "penalties_order"]]
    # price (GBP m) and ownership (%): latest snapshot row per player in the data repo (keyed id -> player_code)
    price, own = pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        ps = pd.read_csv(Path(a.src) / "data" / CURRENT_SEASON / "playerstats.csv", usecols=["id", "gw", "now_cost", "selected_by_percent"])
        ps = ps.sort_values("gw").drop_duplicates("id", keep="last")
        id2code = s.players.set_index("player_id")["player_code"]
        ps["code"] = ps["id"].map(id2code)
        ps = ps.dropna(subset=["code"]).set_index(ps["code"].dropna().astype(int))
        price, own = ps["now_cost"].astype(float), pd.to_numeric(ps["selected_by_percent"], errors="coerce")
    except Exception as e:  # noqa: BLE001
        notes.append(f"Price/ownership snapshot unavailable ({type(e).__name__}).")
    bs = live_bootstrap()
    if bs:
        el = pd.DataFrame(bs["elements"])
        tcode = {t["id"]: t["code"] for t in bs["teams"]}
        st = pd.DataFrame({"player_code": el["code"], "status": el["status"],
                           "chance": el["chance_of_playing_next_round"], "news": el["news"],
                           "penalties_order": el["penalties_order"] if "penalties_order" in el else np.nan})
        tn = el.set_index("code")["team"].map(tcode)
        team_now = tn.where(tn.notna(), team_now.reindex(tn.index)).dropna().astype(int)
        price = (el.set_index("code")["now_cost"] / 10.0)
        own = pd.to_numeric(el.set_index("code")["selected_by_percent"], errors="coerce")
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

    # manager squads listed in config/teams.json (public FPL endpoints; a failure only marks that team unavailable)
    teams_out = []
    tf = ROOT / "config" / "teams.json"
    if tf.exists() and len(price):
        try:
            sp = ps.set_index("code") if "code" in ps.columns else None
            start = (price - pd.to_numeric(ps.set_index(ps["code"].astype(int))["cost_change_start"], errors="coerce").reindex(price.index).fillna(0) / 10.0).round(1).to_dict() if sp is not None else {}
        except Exception:  # noqa: BLE001
            start = {}
        id2code = {int(i): int(c) for i, c in s.players.set_index("player_id")["player_code"].items()}
        for t in json.loads(tf.read_text()):
            try:
                teams_out.append(build_team(int(t["id"]), t.get("label", str(t["id"])), int(gw0), id2code, price.to_dict(), start))
            except Exception as e:  # noqa: BLE001
                teams_out.append({"id": int(t["id"]), "label": t.get("label", str(t["id"])), "error": str(e)[:160]})
                notes.append(f"Could not load team {t['id']} on this run.")
    owned = {x['code'] for t in teams_out for x in t.get('squad', [])}
    # penalty takers: evidence-weighted shares per club, and the expected penalties each player takes over the horizon
    tw, pen_rate, pen_league, pen_conv = ctx.last_pen
    pen_exp = d.groupby("player_code")["lam_pen"].sum() if "lam_pen" in d else pd.Series(dtype=float)
    pens = team_summary(tw, players["web_name"], pen_rate)
    short = s.teams.set_index("code")["short_name"]
    for t in pens:
        t["team"] = str(short.get(t["team"], t["team"]))
    flagged = [t for t in pens if t["status"] in ("conflict",) or any(x["mover"] for x in t["takers"])]
    for t in flagged:
        if t["status"] == "conflict":
            notes.append(f"Penalty watch: {t['team']} - FPL's first-choice taker differs from who has taken the club's recent penalties.")
        for x in t["takers"]:
            if x["mover"]:
                notes.append(f"Penalty watch: {x['name']} ({t['team']}) took {x['prev']} penalties for a previous club; role at the new club unconfirmed.")
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
            "price": float(price[pc]) if pc in price.index and pd.notna(price[pc]) else None,
            "own": float(own[pc]) if pc in own.index and pd.notna(own[pc]) else None,
            "status": stx["status"].get(pc, "a"), "news": str(stx["news"].get(pc) if pd.notna(stx["news"].get(pc)) else "")[:120],
            "ep": [round(float(x), 2) if pd.notna(x) else 0.0 for x in grp["ep"]],
            "p_play": [round(float(x), 2) if pd.notna(x) else 0.0 for x in grp["p_play"]],
            "p60": [round(float(min(1.0, x)), 2) if pd.notna(x) else 0.0 for x in grp["p60_eff"]],
            "pa": [round(float(x), 3) if pd.notna(x) else 0.0 for x in grp["pts_appearance"]],
            "pc": [round(float(x), 3) if pd.notna(x) else 0.0 for x in grp["pts_clean_sheet"]],
            "pr": [round(float(e - a_ - c_), 3) if pd.notna(e) else 0.0 for e, a_, c_ in zip(grp["ep"], grp["pts_appearance"], grp["pts_clean_sheet"])],
            "pst": [round(float(min(1.0, x)), 3) if pd.notna(x) else 0.0 for x in grp["p_start_eff"]],
            "psub": [round(float(min(1.0, x)), 3) if pd.notna(x) else 0.0 for x in grp["p_sub_eff"]],
            "ms": [round(float(x), 0) if pd.notna(x) else 0.0 for x in grp["m_from_start"]],
            "mb": [round(float(x), 0) if pd.notna(x) else 0.0 for x in grp["m_from_sub"]],
            "mins": [round(float(x), 0) if pd.notna(x) else 0.0 for x in grp["m_exp"]],
            "fx": [", ".join(fx.get((pc, gw), [])) or "BLANK" for gw in gws],
            "comp": {k[4:]: round(float(grp["pts_" + k[4:]].iloc[0]), 2) for k in
                     ["pts_appearance", "pts_goals", "pts_assists", "pts_clean_sheet", "pts_goals_conceded", "pts_saves",
                      "pts_defensive", "pts_cards", "pts_bonus"] if "pts_" + k[4:] in grp and pd.notna(grp["pts_" + k[4:]].iloc[0])},
            "sd": round(float(grp["ep_sd"].iloc[0]), 2) if pd.notna(grp["ep_sd"].iloc[0]) else None,
            "low_history": bool(hist.get(pc, 0) < 5),
        }
        if pc in tw.index and (tw.loc[pc, "pen_share"] >= 0.1 or tw.loc[pc, "pen_n_club"] > 0):
            r_ = tw.loc[pc]
            row["pen"] = {"share": round(float(r_["pen_share"]), 2), "taken": int(r_["pen_n_club"]), "fpl": None if pd.isna(r_["fpl_order"]) else int(r_["fpl_order"]),
                         "exp": round(float(pen_exp.get(pc, 0.0)), 2), "mover": bool(r_["mover"])}
        row["total"] = round(float(sum(row["ep"])), 2)
        if row["total"] >= 0.05 or int(pc) in owned:
            out_players.append(row)
    out_players.sort(key=lambda r: -r["total"])
    n_low = sum(r["low_history"] for r in out_players)
    if n_low:
        notes.append(f"{n_low} players have fewer than 5 league appearances on record (flagged low_history).")
    # cheapest price per position across the whole current roster: the compulsory minimum spend per slot
    roster_pos = players["position"]
    floor = {pos: round(float(price.reindex(roster_pos.index[roster_pos == pos]).min()), 1) for pos in SQUAD if len(price)}
    payload = {
        "floor": floor, "squad": SQUAD, "budget": BUDGET, "teams": teams_out,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": CURRENT_SEASON, "current_gw": int(gw0), "horizon": gws, "model_version": params.version,
        "gw_dates": {int(gw): str(s.matches.loc[s.matches["gameweek"] == gw, "kickoff_time"].min()) for gw in gws},
        "notes": notes, "penalties": pens, "players": out_players,
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
