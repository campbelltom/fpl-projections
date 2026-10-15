"""Fixture-level features for the minutes model: how much rest a club has had and will have, and how
lopsided the match is. Built from the club calendar across every competition (league, European
competition, domestic cup), which the data carries for future matches as well as past ones."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400e9
FX_COLS = ["rest_b", "rest_a", "n14", "elo_d", "is_home"]
FX_DEFAULT = {"rest_b": 6.0, "rest_a": 6.0, "n14": 2.0, "elo_d": 0.0, "is_home": 0.5}


def _ns(s: pd.Series) -> np.ndarray:
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_convert(None).to_numpy("datetime64[ns]").astype("int64")


def calendar_map(cal: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for t, g in cal.groupby("team"):
        out[int(t)] = np.sort(_ns(g["date"]))
    return out


def fixture_features(cmap: dict[int, np.ndarray], team: pd.Series, date: pd.Series, elo_d: pd.Series | None,
                     is_home: pd.Series) -> pd.DataFrame:
    """One row per (team, kickoff): days since the previous match, days to the next, matches in the last 14
    days, rating gap (team minus opponent, /400) and home flag."""
    n = len(team)
    d = _ns(date.reset_index(drop=True))
    tm = team.reset_index(drop=True).to_numpy()
    rb, ra, n14 = np.full(n, 10.0), np.full(n, 10.0), np.full(n, 2.0)
    for t in np.unique(tm[~pd.isna(tm)]):
        arr = cmap.get(int(t))
        if arr is None or not len(arr):
            continue
        ix = np.where(tm == t)[0]
        dd = d[ix]
        lo = np.searchsorted(arr, dd, side="left")
        hi = np.searchsorted(arr, dd, side="right")
        prev = np.where(lo > 0, arr[np.maximum(lo - 1, 0)], np.nan)
        nxt = np.where(hi < len(arr), arr[np.minimum(hi, len(arr) - 1)], np.nan)
        rb[ix] = np.clip(np.where(np.isnan(prev), 10.0, (dd - prev) / DAY), 1, 10)
        ra[ix] = np.clip(np.where(np.isnan(nxt), 10.0, (nxt - dd) / DAY), 1, 10)
        n14[ix] = lo - np.searchsorted(arr, dd - int(14 * DAY), side="left")
    out = pd.DataFrame({"rest_b": rb, "rest_a": ra, "n14": n14}, index=team.index)
    out["elo_d"] = (elo_d.reindex(team.index).fillna(0.0).clip(-1.5, 1.5) if elo_d is not None else 0.0)
    out["is_home"] = is_home.reindex(team.index).fillna(0.5).astype(float)
    return out


def rival_features(es: pd.Series, group_key: pd.DataFrame, elig: pd.Series | None = None) -> pd.DataFrame:
    """Competition for a place: the summed starting rate of the other available teammates in his position,
    and the best of them. `es` is each player's own recent starting rate."""
    g = group_key.copy()
    g["es"] = es.to_numpy() * (1.0 if elig is None else elig.reindex(es.index).fillna(False).astype(float).to_numpy())
    cols = list(group_key.columns)
    tot = g.groupby(cols)["es"].transform("sum")
    # best of the others: the group maximum, unless he is it, in which case the second best
    srt = g.sort_values("es", ascending=False)
    top1 = srt.groupby(cols)["es"].transform("first").reindex(g.index)
    top2 = srt.groupby(cols)["es"].transform(lambda s: s.iloc[1] if len(s) > 1 else 0.0).reindex(g.index)
    rv_max = np.where(g["es"] >= top1, top2, top1)
    return pd.DataFrame({"rv_sum": (tot - g["es"]).clip(lower=0), "rv_max": rv_max}, index=es.index)
