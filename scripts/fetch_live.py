"""Live gameweek feed: scores, events and the tracked teams' live points, written to a small public live.json.

Everything comes from the official Fantasy API (bootstrap-static, fixtures, event live, entry picks). Provisional bonus is worked out
from the fixture BPS until the official bonus is awarded, and the automatic substitutions are applied once a team's games are over.
Nothing is written outside a match window, so the last file stays as the final picture of the previous gameweek.
"""
import json, re, sys, time, datetime as dt, urllib.request
from pathlib import Path

API = "https://fantasy.premierleague.com/api"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
EVENT_STATS = [("goals_scored", "g"), ("assists", "a"), ("own_goals", "og"), ("penalties_saved", "ps"),
               ("penalties_missed", "pm"), ("yellow_cards", "yc"), ("red_cards", "rc")]


def fetch(path, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "campbell-fpl/2.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)


def parse_ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def in_window(fixtures, now):
    """True while any match is on, or within five hours of kick-off (so late bonus and the final scores are caught)."""
    for f in fixtures:
        if not f.get("kickoff_time"):
            continue
        ko = parse_ts(f["kickoff_time"])
        if ko <= now <= ko + dt.timedelta(hours=5):
            return True
        if f.get("started") and not f.get("finished"):
            return now - ko < dt.timedelta(days=1)
    return False


def bonus_by_fixture(fixtures):
    """fixture id -> {element: bonus}; the official bonus once it is in, otherwise provisional from BPS."""
    out = {}
    for f in fixtures:
        if not f.get("started"):
            continue
        stats = {s["identifier"]: s for s in f.get("stats", [])}
        official = {x["element"]: x["value"] for side in ("h", "a") for x in stats.get("bonus", {}).get(side, [])}
        if official:
            out[f["id"]] = official
            continue
        bps = [(x["element"], x["value"]) for side in ("h", "a") for x in stats.get("bps", {}).get(side, [])]
        res = {}
        for el, v in bps:
            place = sum(1 for _, w in bps if w > v)
            b = {0: 3, 1: 2, 2: 1}.get(place, 0)
            if b:
                res[el] = b
        out[f["id"]] = res
    return out


def player_points(live, fixtures):
    """element -> (points including bonus, minutes, bps, bonus)."""
    bonus = bonus_by_fixture(fixtures)
    res = {}
    for e in live.get("elements", []):
        pts = bon = 0
        for ex in e.get("explain", []):
            b = bonus.get(ex["fixture"], {}).get(e["id"], 0)
            pts += sum(s["points"] for s in ex["stats"] if s["identifier"] != "bonus") + b
            bon += b
        st = e.get("stats", {})
        res[e["id"]] = (pts, int(st.get("minutes", 0)), int(st.get("bps", 0)), bon)
    return res


def auto_subs(xi, bench, played, done, pos):
    """xi/bench: element ids in pick order. A starter on 0 minutes whose team has finished is replaced by the first bench player
    who can take the place without breaking the formation (a keeper only for a keeper). Returns the new xi and the swaps."""
    xi, swaps, bench = list(xi), [], list(bench)
    for out in [p for p in xi if not played(p) and done(p)]:
        for b in bench:
            if not played(b):
                continue
            if (pos[b] == 1) != (pos[out] == 1):
                continue
            new = [b if p == out else p for p in xi]
            n = lambda k: sum(1 for p in new if pos[p] == k)
            if n(1) == 1 and n(2) >= 3 and n(3) >= 2 and n(4) >= 1:
                xi, swaps = new, swaps + [[out, b]]
                bench.remove(b)
                break
    return xi, swaps


def build_team(label, entry_id, picks, played, done, pts, pos):
    ps = picks["picks"]
    chip = picks.get("active_chip")
    xi = [p["element"] for p in ps if p["position"] <= 11]
    bench = [p["element"] for p in sorted(ps, key=lambda p: p["position"]) if p["position"] > 11]
    mult = {p["element"]: p["multiplier"] for p in ps}
    cap = next((p["element"] for p in ps if p["is_captain"]), None)
    vice = next((p["element"] for p in ps if p["is_vice_captain"]), None)
    swaps = []
    if chip != "bboost":
        xi, swaps = auto_subs(xi, bench, played, done, pos)
        for out, inn in swaps:
            mult[inn], mult[out] = 1, 0
        if cap is not None and not played(cap) and done(cap) and vice is not None and played(vice) and vice in xi:
            mult[vice], mult[cap] = mult[cap], 1      # the vice-captain takes the armband (and the triple, if it is on)
            cap = vice
    else:
        xi = xi + bench
        bench = []
        for p in xi:
            mult[p] = max(mult.get(p, 1), 1)
    rest = [b for b in bench if b not in xi]
    cost = picks.get("entry_history", {}).get("event_transfers_cost", 0)
    rows = lambda ids: [[e, mult.get(e, 0), pts.get(e, (0,))[0], 0 if e not in pts else pts[e][1]] for e in ids]
    total = sum(mult.get(e, 0) * pts.get(e, (0,))[0] for e in xi) - cost
    return {"label": label, "id": entry_id, "chip": chip, "cost": cost, "pts": total, "cap": cap,
            "xi": rows(xi), "bench": rows(rest), "subs": swaps}


def league_table(meta, standings, picks, played, done, pts, pos):
    """Rows for one mini league: points before this gameweek and this gameweek's live points (hits included)."""
    rows = []
    for s in standings:
        pk = picks.get(s["entry"])
        row = {"e": s["entry"], "n": s["entry_name"], "m": s["player_name"], "pre": s["total"] - s["event_total"], "gw": None}
        if pk:
            tm = build_team("", s["entry"], pk, played, done, pts, pos)
            row["gw"] = tm["pts"]
            row["t"] = {k: tm[k] for k in ("xi", "bench", "subs", "cap")}
            row["chip"] = tm["chip"]
            row["left"] = sum(1 for r in tm["xi"] if r[1] and not done(r[0]))
        else:
            row["gw"] = s["event_total"]
        rows.append(row)
    return {"id": meta["id"], "name": meta["name"], "rows": rows}


def build(boot, fixtures, live, picks_by_team, now, leagues=()):
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    el = {e["id"]: e for e in boot["elements"]}
    pos = {i: e["element_type"] for i, e in el.items()}
    pts = player_points(live, fixtures)
    team_fx = {}
    for f in fixtures:
        for t in (f["team_h"], f["team_a"]):
            team_fx.setdefault(t, []).append(f)
    played = lambda e: pts.get(e, (0, 0))[1] > 0
    done = lambda e: all(f.get("finished_provisional") or f.get("finished") for f in team_fx.get(el[e]["team"], [])) if team_fx.get(el[e]["team"]) else False

    fx, need = [], set()
    for f in sorted(fixtures, key=lambda f: f.get("kickoff_time") or ""):
        ev = []
        stats = {s["identifier"]: s for s in f.get("stats", [])}
        for ident, k in EVENT_STATS:
            for side in ("h", "a"):
                for x in stats.get(ident, {}).get(side, []):
                    ev.append([k, x["element"], side, x["value"]])
                    need.add(x["element"])
        fin = bool(f.get("finished_provisional") or f.get("finished"))
        fx.append({"id": f["id"], "ko": f.get("kickoff_time"), "h": teams[f["team_h"]], "a": teams[f["team_a"]],
                   "hs": f.get("team_h_score"), "as": f.get("team_a_score"), "fin": fin, "st": bool(f.get("started")),
                   "m": f.get("minutes") or 0, "ev": ev})
    tms = []
    for label, entry_id, picks in picks_by_team:
        t = build_team(label, entry_id, picks, played, done, pts, pos)
        tms.append(t)
        for r in t["xi"] + t["bench"]:
            need.add(r[0])
    lg = [league_table(m, st, pk, played, done, pts, pos) for m, st, pk in leagues]
    for g in lg:
        for r in g["rows"]:
            for x in (r.get("t") or {}).get("xi", []) + (r.get("t") or {}).get("bench", []):
                need.add(x[0])
    top = sorted(pts, key=lambda e: -pts[e][0])[:15]
    need |= set(top)
    pl = {str(e): [el[e]["web_name"], teams[el[e]["team"]], POS[pos[e]], pts[e][0] if e in pts else 0,
                   pts[e][1] if e in pts else 0, pts[e][3] if e in pts else 0] for e in need if e in el}
    return {"at": now.replace(microsecond=0).isoformat(), "fx": fx, "pl": pl, "teams": tms, "top": top, "lg": lg}


MAX_LEAGUES, MAX_ENTRIES = 3, 60


def league_ids(entry_id, cfg_file="config/leagues.json"):
    """config/leagues.json (a list of ids) wins; otherwise the private classic leagues of the first tracked team."""
    f = Path(cfg_file)
    if f.exists():
        return [int(x) for x in json.loads(f.read_text())][:MAX_LEAGUES]
    ent = fetch(f"entry/{entry_id}/")
    mine = [c for c in ent.get("leagues", {}).get("classic", []) if c.get("league_type") == "x"]
    mine.sort(key=lambda c: 0 if re.search(r"horses\s+and\s+ants", c.get("name", ""), re.I) else 1)   # that league first
    return [c["id"] for c in mine][:MAX_LEAGUES]


def fetch_leagues(gw, tracked_picks, teams_cfg):
    from concurrent.futures import ThreadPoolExecutor
    out = []
    try:
        ids = league_ids(teams_cfg[0]["id"])
    except Exception as e:  # noqa: BLE001
        print(f"no leagues: {e}")
        return out
    have = {eid: pk for _, eid, pk in tracked_picks}
    cache = dict(have)

    def get_picks(eid):
        if eid in cache:
            return eid, cache[eid]
        try:
            return eid, fetch(f"entry/{eid}/event/{gw}/picks/", tries=2)
        except Exception:  # noqa: BLE001
            return eid, None
    for lid in ids:
        try:
            st = fetch(f"leagues-classic/{lid}/standings/")
        except Exception as e:  # noqa: BLE001
            print(f"league {lid} not available: {e}")
            continue
        res = st["standings"]["results"][:MAX_ENTRIES]
        need = [r["entry"] for r in res if r["entry"] not in cache]
        with ThreadPoolExecutor(6) as ex:
            for eid, pk in ex.map(get_picks, need):
                cache[eid] = pk
        out.append(({"id": lid, "name": st["league"]["name"]}, res, {r["entry"]: cache.get(r["entry"]) for r in res}))
    return out


def main(dest, teams_file="config/teams.json"):
    now = dt.datetime.now(dt.timezone.utc)
    boot = fetch("bootstrap-static/")
    events = boot["events"]
    cur = next((e for e in events if e.get("is_current")), None)
    cands = [e for e in events if cur and e["id"] in (cur["id"], cur["id"] + 1)] or events[:1]
    for ev in cands:
        fixtures = fetch(f"fixtures/?event={ev['id']}")
        if in_window(fixtures, now):
            break
    else:
        print("no match window: nothing to publish")
        return False
    gw = ev["id"]
    live = fetch(f"event/{gw}/live/")
    picks = []
    teams_cfg = json.loads(Path(teams_file).read_text())
    for t in teams_cfg:
        try:
            picks.append((t["label"], t["id"], fetch(f"entry/{t['id']}/event/{gw}/picks/")))
        except Exception as e:  # noqa: BLE001
            print(f"picks for {t['label']} not available: {e}")
    leagues = fetch_leagues(gw, picks, teams_cfg)
    res = build(boot, fixtures, live, picks, now, leagues)
    res["gw"] = gw
    Path(dest).write_text(json.dumps(res, separators=(",", ":")))
    print(f"GW{gw}: {len(res['fx'])} fixtures, {len(res['teams'])} teams -> {dest}")
    return True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "live.json")
