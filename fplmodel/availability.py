"""Availability: how likely a player is to be fit/selectable in each upcoming gameweek.

Calibrated from the history of FPL status flags (a=available, d=doubtful with a chance, i=injured,
s=suspended, u=unavailable, n=not in squad).  The status stored in a gameweek's row is the state
*before* that gameweek's matches, so for the next gameweek it is simply the known state (k=0) and for
later gameweeks (k>=1) we use the empirical probability that a player in that state is fit k gameweeks
later.  A dated return ("Expected back 7 Nov") is used when present, discounted by how often such
dates turn out to be right.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
RET_RE = re.compile(r"expected back (\d{1,2}) ([A-Za-z]{3})", re.I)
HORIZON = 8


def v_of(status, chance) -> float:
    """FPL-coded availability in [0,1]."""
    if status == "a":
        return 1.0
    if status == "d":
        c = 50.0 if chance is None or (isinstance(chance, float) and np.isnan(chance)) else float(chance)
        return c / 100.0
    return 0.0


def group_of(status, chance) -> str:
    if status == "a":
        return "a"
    if status == "d":
        c = 50 if chance is None or (isinstance(chance, float) and np.isnan(chance)) else int(chance)
        return f"d{c}"
    if status in ("i", "s"):
        return status
    return "u"


def parse_return(news, ref: pd.Timestamp) -> pd.Timestamp | None:
    if not isinstance(news, str):
        return None
    m = RET_RE.search(news)
    if not m:
        return None
    mon = MONTHS.get(m.group(2).lower())
    if not mon:
        return None
    day = int(m.group(1))
    for yr in (ref.year, ref.year + 1):
        try:
            d = pd.Timestamp(year=yr, month=mon, day=day, tz="UTC")
        except ValueError:
            return None
        if d >= ref - pd.Timedelta(days=45):
            return d
    return None


@dataclass
class AvailabilityModel:
    trans: dict = field(default_factory=dict)      # (group, k) -> E[v_{g+k}]
    v_returned: float = 0.8                        # P(fit) once a stated return date has passed
    n: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, status_table: pd.DataFrame, max_gidx: int) -> "AvailabilityModel":
        """status_table: player_code, season_idx, gw, gidx, status, chance, news, gw_date.
        Only pairs whose *later* gameweek has gidx <= max_gidx are used (no look-ahead)."""
        st = status_table.dropna(subset=["status"]).copy()
        st["v"] = [v_of(s, c) for s, c in zip(st["status"], st["chance"])]
        st["grp"] = [group_of(s, c) for s, c in zip(st["status"], st["chance"])]
        key = st[["player_code", "season_idx", "gw", "v"]]
        trans, cnt = {}, {}
        for k in range(1, HORIZON + 1):
            nxt = key.assign(gw=key["gw"] - k).rename(columns={"v": "v_next"})
            j = st.merge(nxt, on=["player_code", "season_idx", "gw"], how="inner")
            j = j[j["gidx"] + k <= max_gidx]
            g = j.groupby("grp")["v_next"].agg(["mean", "size"])
            for grp, r in g.iterrows():
                trans[(grp, k)] = float(r["mean"]); cnt[(grp, k)] = int(r["size"])
        # reliability of dated returns: P(fit) in the GWs whose date is after the stated return
        rows = []
        inj = st[st["status"].isin(["i", "d"]) & st["news"].notna()]
        gwd = status_table.drop_duplicates(["season_idx", "gw"]).set_index(["season_idx", "gw"])["gw_date"]
        fut = st[["player_code", "season_idx", "gw", "v"]].rename(columns={"gw": "gw2", "v": "v2"})
        for k in range(1, HORIZON + 1):
            j = inj.assign(gw2=inj["gw"] + k).merge(fut, on=["player_code", "season_idx", "gw2"], how="inner")
            j = j[j["gidx"] + k <= max_gidx]
            for r in j.itertuples():
                ret = parse_return(r.news, r.gw_date)
                d2 = gwd.get((r.season_idx, r.gw2))
                if ret is not None and d2 is not None and d2 >= ret:
                    rows.append(r.v2)
        vret = float(np.mean(rows)) if len(rows) >= 30 else 0.8
        return cls(trans, vret, cnt)

    def _t(self, grp: str, k: int) -> float:
        k = int(min(max(k, 1), HORIZON))
        if (grp, k) in self.trans and self.n.get((grp, k), 0) >= 15:
            return self.trans[(grp, k)]
        base = {"a": 0.97, "i": 0.35, "s": 0.55, "u": 0.0}.get(grp[0] if grp[0] != "d" else "a", 0.5)
        if grp.startswith("d"):
            return float(np.clip(0.55 + 0.4 * int(grp[1:]) / 100, 0.4, 0.97))
        return base

    def curve(self, status, chance, news, gw_dates: list[pd.Timestamp], now: pd.Timestamp) -> np.ndarray:
        """Availability factor for each upcoming gameweek (list of the gameweek's first kickoff dates)."""
        grp = group_of(status, chance)
        out = np.zeros(len(gw_dates))
        ret = parse_return(news, now) if status in ("i", "d", "s") else None
        for j, d in enumerate(gw_dates):
            if grp == "u":
                out[j] = 0.0
            elif j == 0:
                out[j] = v_of(status, chance) if grp[0] != "i" else 0.02
            else:
                v = self._t(grp, j)
                if ret is not None and status in ("i", "s"):
                    v = self.v_returned if d >= ret else 0.03
                out[j] = v
        return np.clip(out, 0.0, 1.0)
