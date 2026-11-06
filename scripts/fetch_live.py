"""Live gameweek feed: scores, events and the tracked teams' live points, written to a small public live.json.

Everything comes from the official Fantasy API (bootstrap-static, fixtures, event live, entry picks). Provisional bonus is worked out
from the fixture BPS until the official bonus is awarded, and the automatic substitutions are applied once a team's games are over.
Nothing is written outside a match window, so the last file stays as the final picture of the previous gameweek.
"""
import json, sys, time, datetime as dt, urllib.request
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


def build(boot, fixtures, live, picks_by_team, now):
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
    top = sorted(pts, key=lambda e: -pts[e][0])[:15]
    need |= set(top)
    pl = {str(e): [el[e]["web_name"], teams[el[e]["team"]], POS[pos[e]], pts[e][0] if e in pts else 0,
                   pts[e][1] if e in pts else 0, pts[e][3] if e in pts else 0] for e in need if e in el}
    return {"at": now.replace(microsecond=0).isoformat(), "fx": fx, "pl": pl, "teams": tms, "top": top}


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
    for t in json.loads(Path(teams_file).read_text()):
        try:
            picks.append((t["label"], t["id"], fetch(f"entry/{t['id']}/event/{gw}/picks/")))
        except Exception as e:  # noqa: BLE001
            print(f"picks for {t['label']} not available: {e}")
    res = build(boot, fixtures, live, picks, now)
    res["gw"] = gw
    Path(dest).write_text(json.dumps(res, separators=(",", ":")))
    print(f"GW{gw}: {len(res['fx'])} fixtures, {len(res['teams'])} teams -> {dest}")
    return True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "live.json")
