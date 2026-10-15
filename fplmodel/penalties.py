"""Penalty model: who takes a team's penalties, and how many penalties a team gets.

Penalties are the one scoring event a single player can monopolise, and the evidence about who takes
them is specific to a club. So this module keeps them out of the general goal-share model:

  team rate     penalties awarded per match, decayed history, shrunk to the league rate
  taker weight  w_i = decayed penalties taken FOR THE CURRENT CLUB
                    + a prior from FPL's own `penalties_order` (1st choice counts most)
                    + a small spread by xG share so nobody in the squad is exactly zero
  transfers     a player's penalties at a previous club do not count at his new one. A taker who has
                moved gets only the FPL-order prior (plus a small bump if he took several before) and
                is flagged as unconfirmed until he takes one for the new club.

A penalty is taken by whoever is designated *and on the pitch*, so at prediction time the weights are
multiplied by the expected fraction of the match each player is on the field and re-normalised within
the team (see predict.py): when the taker is injured the next in line inherits the penalties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PEN_CONV = 0.80          # league conversion of penalties; refreshed from data in team_rates()
PEN_XG = 0.78            # xG that FPL-data providers assign a penalty (used to split open-play xG)

# tunables (evaluated by scripts/eval_penalties.py)
PRIOR_ORDER = {1: 2.0, 2: 0.6, 3: 0.25}     # pseudo-penalties given to FPL's 1st/2nd/3rd listed taker
HL_PEN = 30.0                                # half-life of penalty evidence, in the club's matches
SPREAD = 0.5                                 # total pseudo-penalties spread over the squad by xG share
MOVER_BUMP = 0.6                             # extra prior for a recent mover who took >= 2 for his old club
RATE_PRIOR_MATCHES = 25.0                    # pseudo-matches pulling a team's penalty rate to the league rate


def add_pen_columns(pms: pd.DataFrame) -> pd.DataFrame:
    """Per player-match penalty counts and open-play xG (pms already holds scored/missed)."""
    out = pms.copy()
    sc = out["penalties_scored"].fillna(0.0) if "penalties_scored" in out else 0.0
    ms = out["penalties_missed"].fillna(0.0) if "penalties_missed" in out else 0.0
    out["pen_sc"], out["pen_ms"] = sc, ms
    out["pen_taken"] = out["pen_sc"] + out["pen_ms"]
    return out


def team_rates(opp: pd.DataFrame, asof: pd.Timestamp, season_idx_now: int, team_idx_asof: dict[int, int],
               season_w: float = 0.7) -> tuple[pd.Series, float, float]:
    """Penalties awarded per match for each team, and the league rate / conversion."""
    a = opp[opp["date"] < asof]
    per = a.groupby(["match_id", "team"]).agg(pens=("pen_taken", "sum"), conv_n=("pen_sc", "sum"),
                                              idx=("team_match_idx", "first"), sidx=("season_idx", "first")).reset_index()
    league = float(per["pens"].mean()) if len(per) else 0.11
    tot = float(per["pens"].sum())
    conv = float(per["conv_n"].sum() / tot) if tot >= 40 else PEN_CONV
    age = np.clip(per["team"].map(team_idx_asof).fillna(0).to_numpy() - per["idx"].to_numpy() - 1, 0, None)
    w = (0.5 ** (age / 60.0)) * season_w ** np.clip(season_idx_now - per["sidx"].to_numpy(), 0, None)
    g = pd.DataFrame({"team": per["team"], "n": per["pens"] * w, "d": w}).groupby("team").sum()
    rate = (g["n"] + RATE_PRIOR_MATCHES * league) / (g["d"] + RATE_PRIOR_MATCHES)
    return rate, league, conv


def taker_weights(opp: pd.DataFrame, asof: pd.Timestamp, season_idx_now: int, team_idx_asof: dict[int, int],
                  team_now: pd.Series, fpl_order: pd.Series, xg_share: pd.Series, season_w: float = 0.7,
                  prior_order: dict | None = None, hl: float = HL_PEN, spread: float = SPREAD,
                  mover_bump: float = MOVER_BUMP) -> pd.DataFrame:
    """One row per player in `team_now` (player_code -> current team code).

    Columns: team_code, pen_w (weight), pen_share (w / team total), pen_ev (decayed pens taken for the
    club), pen_n_club (raw count for the club in the data window), pen_last_gw, pen_n_prev (taken for
    other clubs), fpl_order, club_matches (matches in the data for the current club)."""
    prior_order = PRIOR_ORDER if prior_order is None else prior_order
    a = opp[opp["date"] < asof]
    tn = team_now.rename("team_now")
    a = a.merge(tn, left_on="player_code", right_index=True, how="inner")
    here = a["team"] == a["team_now"]
    age = np.clip(a["team"].map(team_idx_asof).fillna(0).to_numpy() - a["team_match_idx"].to_numpy() - 1, 0, None)
    w = (0.5 ** (age / hl)) * season_w ** np.clip(season_idx_now - a["season_idx"].to_numpy(), 0, None)

    ev = pd.Series((a["pen_taken"].to_numpy() * w * here.to_numpy())).groupby(a["player_code"].to_numpy()).sum()
    n_club = pd.Series((a["pen_taken"].to_numpy() * here.to_numpy())).groupby(a["player_code"].to_numpy()).sum()
    n_prev = pd.Series((a["pen_taken"].to_numpy() * (~here).to_numpy())).groupby(a["player_code"].to_numpy()).sum()
    played_here = ((a["minutes"] > 0) & here)
    club_matches = played_here.groupby(a["player_code"]).sum()
    last_gw = a[here & (a["pen_taken"] > 0)].groupby("player_code").agg(g=("gw", "max"), s=("season", "max"))

    d = pd.DataFrame(index=team_now.index)
    d["team_code"] = team_now
    d["pen_ev"] = ev.reindex(d.index).fillna(0.0)
    d["pen_n_club"] = n_club.reindex(d.index).fillna(0.0)
    d["pen_n_prev"] = n_prev.reindex(d.index).fillna(0.0)
    d["club_matches"] = club_matches.reindex(d.index).fillna(0.0)
    d["pen_last_gw"] = last_gw["g"].reindex(d.index)
    d["pen_last_season"] = last_gw["s"].reindex(d.index)
    d["fpl_order"] = fpl_order.reindex(d.index)
    prior = d["fpl_order"].map(lambda o: prior_order.get(int(o), 0.0) if pd.notna(o) else 0.0)
    xs = xg_share.reindex(d.index).fillna(0.0).clip(lower=0)
    tot = xs.groupby(d["team_code"]).transform("sum").replace(0, np.nan)
    sp = (spread * xs / tot).fillna(0.0)
    mover = ((d["pen_n_prev"] >= 2) & (d["club_matches"] < 15) & (d["pen_ev"] < 0.5)).astype(float)
    d["mover"] = mover > 0
    # a mover only matters while the club has no established taker: fade the bump as club evidence builds
    club_ev = d["pen_ev"].groupby(d["team_code"]).transform("sum")
    bump = mover_bump * mover * np.clip(1.0 - club_ev / 4.0, 0.0, 1.0)
    d["pen_w"] = d["pen_ev"] + prior + sp + bump
    d["pen_share"] = d["pen_w"] / d["pen_w"].groupby(d["team_code"]).transform("sum").replace(0, np.nan)
    d["pen_share"] = d["pen_share"].fillna(0.0)
    return d


def team_summary(tw: pd.DataFrame, names: pd.Series, rate: pd.Series) -> list[dict]:
    """Per team: the takers ranked by weight, with the evidence and a status flag for the site.

    status: confirmed   the top-weighted player has taken penalties for this club and FPL agrees
            fpl-only    FPL lists a taker but nobody has taken one for this club (new club or no pens yet)
            conflict    FPL's first choice is not the player with the most recent evidence
            unclear     no first choice listed and no evidence
    """
    out = []
    for t, g in tw.groupby("team_code"):
        g = g.sort_values("pen_w", ascending=False)
        top = g.iloc[0]
        tot_ev = float(g["pen_n_club"].sum())
        fpl1 = g[g["fpl_order"] == 1]
        ev_lead = g.sort_values(["pen_ev", "pen_n_club"], ascending=False).iloc[0] if g["pen_ev"].max() > 0 else None
        if ev_lead is None:
            status = "fpl-only" if len(fpl1) else "unclear"
        elif len(fpl1) and fpl1.index[0] != ev_lead.name and ev_lead["pen_ev"] > 1.5 * float(fpl1.iloc[0]["pen_ev"]) + 0.2:
            status = "conflict"
        elif len(fpl1) == 0:
            status = "unclear"
        else:
            status = "confirmed"
        rows = []
        for pc, r in g[(g["pen_w"] > 0.35) | (g["fpl_order"].notna()) | (g["pen_n_club"] > 0)].head(5).iterrows():
            rows.append({"code": int(pc), "name": str(names.get(pc, pc)), "share": round(float(r["pen_share"]), 3),
                         "taken": int(r["pen_n_club"]), "fpl": None if pd.isna(r["fpl_order"]) else int(r["fpl_order"]),
                         "last_gw": None if pd.isna(r["pen_last_gw"]) else int(r["pen_last_gw"]),
                         "mover": bool(r["mover"]), "prev": int(r["pen_n_prev"]), "matches": int(r["club_matches"])})
        out.append({"team": int(t), "rate": round(float(rate.get(t, np.nan)), 3), "status": status, "total": int(tot_ev),
                    "takers": rows})
    return out
