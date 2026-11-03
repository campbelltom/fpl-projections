"""Yellow-card suspension risk: a player who reaches a card threshold (5 by GW19, 10 by GW32, 15 by GW37) misses the next
match(es). The chance of that is worked out from his card rate and expected minutes, and the affected gameweeks are scaled down."""
from __future__ import annotations

import math

import pandas as pd

# (cards, last gameweek the cards count in, matches banned)
YC_RULES = [(5, 19, 1), (10, 32, 2), (15, 37, 3)]
SCALE = ["ep", "p_play", "p60_eff", "p_start_eff", "p_sub_eff", "m_from_start", "m_from_sub", "m_exp", "lam_g", "lam_a", "e_saves"]


def reach_probs(lams: dict, left: int) -> dict:
    """lams: gw -> expected yellows that gameweek. Returns gw -> probability the threshold is crossed during that gameweek."""
    state = [1.0] + [0.0] * (left - 1)          # P(cards gained so far = k), k < left
    out = {}
    for g in sorted(lams):
        lam = lams[g]
        pmf = [math.exp(-lam) * lam ** j / math.factorial(j) for j in range(left + 1)]
        new, crossed = [0.0] * left, 0.0
        for k, pk in enumerate(state):
            if pk == 0:
                continue
            for j in range(left - k):
                new[k + j] += pk * pmf[j]
            crossed += pk * max(0.0, 1.0 - sum(pmf[: left - k]))
        out[g], state = crossed, new
    return out


def ban_risk(d: pd.DataFrame, yc: pd.Series, rules=YC_RULES) -> dict:
    """(player_code, gw) -> probability he is serving a card suspension in that gameweek."""
    risk: dict = {}
    if not len(yc) or "yc90" not in d:
        return risk
    sub = d[d["player_code"].isin(yc[yc >= 1].index)]
    for pc, grp in sub.groupby("player_code"):
        y = int(yc[pc])
        rule = next(((n, last, ban) for n, last, ban in rules if y < n), None)
        if rule is None or rule[0] - y > 2:
            continue
        n, last, ban = rule
        lam = (grp["yc90"] * grp["m_exp"] / 90.0).groupby(grp["gw"]).sum()
        lam = {int(g): float(v) for g, v in lam.items() if g <= last and v > 0}
        if not lam:
            continue
        gws = sorted(set(int(g) for g in grp["gw"]))
        for g, p in reach_probs(lam, n - y).items():
            nxt = [h for h in gws if h > g][:ban]
            for h in nxt:
                risk[(int(pc), h)] = min(1.0, risk.get((int(pc), h), 0.0) + p)
    return risk


def apply_ban_risk(d: pd.DataFrame, risk: dict) -> pd.DataFrame:
    if not risk:
        return d
    d = d.copy()
    f = pd.Series([1.0 - risk.get((int(p), int(g)), 0.0) for p, g in zip(d["player_code"], d["gw"])], index=d.index)
    for c in [c for c in d.columns if c.startswith("pts_")] + [c for c in SCALE if c in d]:
        d[c] = d[c] * f
    return d
