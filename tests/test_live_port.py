"""The page's JavaScript scoring (site/index.html, between LIVE-CALC markers) must give exactly what scripts/fetch_live.py gives."""
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_live as L  # noqa: E402


def scenario(seed):
    random.seed(seed)
    teams = [{"id": i, "short_name": n} for i, n in enumerate(["ARS", "AVL", "MCI", "TOT"], 1)]
    els, eid = [], 1
    for t in range(1, 5):
        for k, ty in enumerate([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]):
            els.append({"id": eid, "web_name": f"P{eid}", "team": t, "element_type": ty})
            eid += 1
    st = lambda i, h, a: {"identifier": i, "h": [{"element": e, "value": v} for e, v in h], "a": [{"element": e, "value": v} for e, v in a]}
    fx = [
        {"id": 1, "team_h": 1, "team_a": 2, "started": True, "finished": False, "finished_provisional": False, "minutes": 67, "kickoff_time": "2026-11-07T15:00:00Z", "team_h_score": 2, "team_a_score": 1,
         "stats": [st("goals_scored", [(13, 2)], [(28, 1)]), st("assists", [(9, 1)], []), st("yellow_cards", [], [(18, 1)]),
                   st("bps", [(13, 45), (9, 30), (3, 30), (4, 25)], [(28, 33), (17, 20)])]},
        {"id": 2, "team_h": 3, "team_a": 4, "started": True, "finished": True, "finished_provisional": True, "minutes": 90, "kickoff_time": "2026-11-07T12:30:00Z", "team_h_score": 0, "team_a_score": 0,
         "stats": [st("bonus", [(33, 3)], [(50, 1)])]},
    ]
    live = {"elements": []}
    for e in els:
        mins = random.choice([0, 0, 60, 90])
        ex = [{"fixture": 1 if e["team"] < 3 else 2, "stats": [{"identifier": "minutes", "points": 2 if mins else 0, "value": mins}, {"identifier": "bonus", "points": 0, "value": 0}]}] if mins else []
        live["elements"].append({"id": e["id"], "stats": {"minutes": mins, "bps": random.randint(0, 40)}, "explain": ex})

    def picks(chip=None, cap=1, vice=2):
        ids = random.sample(range(1, 61), 15)
        gk = [i for i in ids if els[i - 1]["element_type"] == 1]
        # a legal-ish 15: just use whatever positions turn up; the two ports must agree either way
        return {"active_chip": chip, "entry_history": {"event_transfers_cost": random.choice([0, 4])},
                "picks": [{"element": e, "position": i + 1, "multiplier": 2 if i == 0 else 1 if i < 11 else 0, "is_captain": i == 0, "is_vice_captain": i == 1} for i, e in enumerate(ids)]}
    pk = {100 + i: picks("bboost" if i == 3 else None) for i in range(6)}
    standings = [{"entry": 100 + i, "entry_name": f"T{i}", "player_name": f"M{i}", "total": 300 + i * 7, "event_total": 20 + i} for i in range(7)]
    return teams, els, fx, live, pk, standings


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("seed", range(12))
def test_js_matches_python(seed, tmp_path):
    teams, els, fx, live, pk, standings = scenario(seed)
    now = dt.datetime(2026, 11, 7, 16, 20, tzinfo=dt.timezone.utc)
    boot = {"teams": teams, "elements": els}
    tracked = [("A", 100, pk[100]), ("B", 103, pk[103])]
    lg = [({"id": 1, "name": "L"}, standings, {s["entry"]: pk.get(s["entry"]) for s in standings})]
    want = L.build(boot, fx, live, tracked, now, lg)
    html = (ROOT / "site" / "index.html").read_text()
    code = re.search(r"/\* LIVE-CALC-START.*?\*/(.*?)/\* LIVE-CALC-END \*/", html, re.S).group(1)
    (tmp_path / "in.json").write_text(json.dumps({"boot": boot, "fx": fx, "live": live, "tracked": tracked,
                                                  "lg": [[lg[0][0], standings, [[k, v] for k, v in lg[0][2].items()]]], "now": want["at"]}))
    (tmp_path / "run.js").write_text(code + """
const d = JSON.parse(require('fs').readFileSync(process.argv[2]));
const lg = d.lg.map(([m, st, pk]) => [m, st, new Map(pk.map(([k, v]) => [k, v]))]);
console.log(JSON.stringify(lvBuild(d.boot, d.fx, d.live, d.tracked, d.now, lg)));
""")
    got = json.loads(subprocess.run(["node", str(tmp_path / "run.js"), str(tmp_path / "in.json")], capture_output=True, text=True, check=True).stdout)
    want = json.loads(json.dumps(want))
    for t in want["teams"] + [r for g in want["lg"] for r in g["rows"]]:
        pass
    assert got == want
