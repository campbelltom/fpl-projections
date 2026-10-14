"""Walk-forward backtest: for every past gameweek, rebuild the model using only data available before
that gameweek's first kickoff, predict, and compare with the points players actually scored."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .availability import AvailabilityModel
from .config import Params
from .engine import Context
from .predict import BonusModel, aggregate_gameweek
from .scoring import Rules

log = logging.getLogger("fplmodel.backtest")


def walk_forward(ctx: Context, params: Params, rules: Rules, plan: list[tuple[str, int]], bonus: BonusModel | None = None,
                 horizon: int = 1) -> pd.DataFrame:
    """plan: list of (season_name, gw). Returns player x gw predictions with actuals attached."""
    out = []
    for season, gw in plan:
        si = ctx.names.index(season)
        s = ctx.seasons[season]
        gd = ctx.gw_date[(ctx.gw_date["season"] == season) & (ctx.gw_date["gw"] == gw)]
        if gd.empty:
            continue
        asof = gd["gw_date"].iloc[0] - pd.Timedelta(minutes=1)
        # who is registered to which team this gameweek, and their pre-match status
        reg = s.team_hist[s.team_hist["gw"] == gw].set_index("player_code")["team_code"]
        # status rows are post-gameweek snapshots: what is known before GW g is row g-1
        st = ctx.status[(ctx.status["season"] == season) & (ctx.status["gw"] == gw - 1)][["player_code", "status", "chance", "news"]]
        gws = list(range(gw, gw + horizon))
        am = AvailabilityModel.fit(ctx.status, si * 100 + gw - 1)
        d = ctx.predict(asof, season, gws, params, rules, st, bonus, team_now=reg, avail_model=am)
        g = aggregate_gameweek(d)
        g["season"] = season
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    act = ctx.actual.copy()
    res = res.merge(act.drop(columns=["ep_next", "ep_this"], errors="ignore"), on=["player_code", "season", "gw"], how="left")
    # FPL's own pre-deadline expectation for GW g is `ep_next` from row g-1. (Row g's ep_next/form already
    # contain GW g's points: a leaked benchmark, kept only as `ep_next_leaky` for reference.)
    legit = act[["player_code", "season", "gw", "ep_next"]].copy()
    legit["gw"] = legit["gw"] + 1
    res = res.merge(legit, on=["player_code", "season", "gw"], how="left")
    res = res.merge(act[["player_code", "season", "gw", "ep_next"]].rename(columns={"ep_next": "ep_next_leaky"}),
                    on=["player_code", "season", "gw"], how="left")
    pos = pd.concat([s.players.set_index("player_code")["position"] for s in ctx.seasons.values()]).groupby(level=0).last()
    res["position"] = res["player_code"].map(pos)
    return res


def metrics(df: pd.DataFrame, pred="ep", actual="event_points") -> dict:
    d = df.dropna(subset=[pred, actual])
    e = d[pred] - d[actual]
    out = {"n": int(len(d)), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}
    if len(d) > 10:
        out["spearman"] = float(d[pred].rank().corr(d[actual].rank()))
    return out
