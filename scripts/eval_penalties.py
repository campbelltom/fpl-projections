"""How well does the taker model predict who takes the next penalty?   python scripts/eval_penalties.py <source_dir>

For every penalty in 2025-26 (from GW 4) and 2026-27, rebuild the taker weights using only what was known
before that match (club history + FPL's penalties_order from the previous gameweek) and score the
probability given to the player who actually took it, among the players who featured for the team."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel import penalties as P  # noqa: E402
from fplmodel.config import Params  # noqa: E402
from fplmodel.engine import Context  # noqa: E402

ctx = Context.build(sys.argv[1])
opp = ctx.opp
names = ctx.names
ev = opp[(opp["pen_taken"] > 0) & (((opp["season"] == "2025-2026") & (opp["gw"] >= 4)) | (opp["season"] == "2026-2027"))]
ev = ev.sort_values("date")
print("penalty events:", len(ev), "matches", ev["match_id"].nunique())
st = ctx.status[["season", "gw", "player_code", "penalties_order"]]
tm_idx_cache = {}


def team_idx(asof):
    k = asof
    if k not in tm_idx_cache:
        tm_idx_cache[k] = ctx.tm[ctx.tm["date"] < asof].groupby("team").size().to_dict()
    return tm_idx_cache[k]


rows = []
for (mid, team), g in ev.groupby(["match_id", "team"]):
    r0 = g.iloc[0]
    asof = r0["date"] - pd.Timedelta(minutes=1)
    si = names.index(r0["season"])
    reg = opp[(opp["match_id"] == mid) & (opp["team"] == team) & (opp["minutes"] > 0)]
    team_now = pd.Series(team, index=reg["player_code"].unique())
    fo = st[(st["season"] == r0["season"]) & (st["gw"] == r0["gw"] - 1)].drop_duplicates("player_code").set_index("player_code")["penalties_order"]
    a = opp[(opp["date"] < asof) & (opp["season_idx"] >= si - 1)]
    xs = (a.groupby("player_code")["xg_open"].sum() / a.groupby("player_code")["minutes"].sum().clip(lower=90)).reindex(team_now.index).fillna(0)
    rows.append((mid, team, asof, si, team_now, fo, xs, g.set_index("player_code")["pen_taken"]))
print("event groups:", len(rows))


def score(**kw):
    ll, hit1 = [], 0
    for mid, team, asof, si, team_now, fo, xs, taken in rows:
        fo_ = fo.reindex(team_now.index)
        tw = P.taker_weights(opp, asof, si, team_idx(asof), team_now, fo_, xs, 0.7, **kw)
        w = tw["pen_w"]
        if w.sum() <= 0:
            continue
        pr = w / w.sum()
        for pc, n in taken.items():
            if pc in pr.index:
                ll.append(n * np.log(max(pr[pc], 1e-4)))
        top = pr.idxmax()
        hit1 += int(top in taken.index)
    return float(np.sum(ll) / max(len(ll), 1)), hit1 / len(rows)


base = score(prior_order={}, spread=1.0, hl=1e-9)       # xG-share spread only (no history)
print(f"spread only                      mean log-lik {base[0]:.3f}  top-1 hit {base[1]:.2f}")
for label, kw in [
    ("history only", dict(prior_order={}, spread=0.5)),
    ("FPL order only", dict(prior_order={1: 2.0, 2: 0.6, 3: 0.25}, spread=0.5, hl=1e-9)),
]:
    s = score(**kw)
    print(f"{label:32s} mean log-lik {s[0]:.3f}  top-1 hit {s[1]:.2f}")
best = None
for hl in (15, 30, 60, 120):
    for p1 in (0.5, 1.0, 2.0, 4.0):
        for sp in (0.2, 0.5, 1.0):
            s = score(prior_order={1: p1, 2: p1 * 0.3, 3: p1 * 0.12}, spread=sp, hl=hl)
            if best is None or s[0] > best[0][0]:
                best = (s, hl, p1, sp)
            print(f"  hl {hl:>3} p1 {p1:>3} spread {sp:>3}: ll {s[0]:.3f} top1 {s[1]:.2f}")
print("BEST", best)
