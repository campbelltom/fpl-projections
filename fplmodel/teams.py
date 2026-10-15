"""Team-level model.

For each team statistic s (xG, shots on target, defensive actions) we fit

    E[ s_{team t vs opp o, venue} ] = exp( mu + h * is_home + a_t + d_o )

by recency-weighted, ridge-penalised Poisson regression.  a_t is how much of the statistic team t
produces, d_o is how much opponent o concedes of it.  The same fitted factors are used in both
directions:

  * backwards - a player's historical output is expressed as a *share of his team's output*, so it
    is already stripped of the opposition and venue he happened to face;
  * forwards  - the expected team output for an upcoming fixture is exp(mu + h + a_t + d_o).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Params
from .data import Season

TEAM_STATS = ["xg", "sot", "cbit"]
# Log-factor priors for clubs with no top-flight data in the window (promoted sides). Re-estimated
# from the data by scripts/estimate_promoted_prior.py and stored in config/tuned.json.
DEFAULT_PROMOTED_PRIOR = {"xg": (-0.14, 0.10), "sot": (-0.12, 0.08), "cbit": (0.04, -0.03)}


def build_team_matches(seasons: dict[str, Season]) -> pd.DataFrame:
    """One row per team per finished match, with 'for' and 'against' versions of each statistic."""
    rows = []
    for name, s in seasons.items():
        m = s.matches[s.matches["finished"]].copy()
        if m.empty:
            continue
        cb = s.pms.groupby(["match_id", "team_code"])["cbit"].sum().rename("cbit").reset_index()
        cbh = cb.rename(columns={"team_code": "home_code", "cbit": "home_cbit"})
        cba = cb.rename(columns={"team_code": "away_code", "cbit": "away_cbit"})
        m = m.merge(cbh, on=["match_id", "home_code"], how="left").merge(cba, on=["match_id", "away_code"], how="left")
        for side, other in (("home", "away"), ("away", "home")):
            rows.append(pd.DataFrame({
                "season": name, "gw": m["gameweek"], "match_id": m["match_id"], "date": m["kickoff_time"],
                "team": m[f"{side}_code"], "opp": m[f"{other}_code"], "is_home": int(side == "home"),
                "goals_for": m[f"{side}_score"], "goals_against": m[f"{other}_score"],
                "xg": m[f"{side}_expected_goals_xg"], "xg_against": m[f"{other}_expected_goals_xg"],
                "sot": m[f"{side}_shots_on_target"], "sot_against": m[f"{other}_shots_on_target"],
                "shots": m[f"{side}_total_shots"], "shots_against": m[f"{other}_total_shots"],
                "cbit": m[f"{side}_cbit"], "cbit_against": m[f"{other}_cbit"],
                "elo_t": pd.to_numeric(m.get(f"{side}_team_elo"), errors="coerce") if f"{side}_team_elo" in m else np.nan,
                "elo_d": ((pd.to_numeric(m.get(f"{side}_team_elo"), errors="coerce") - pd.to_numeric(m.get(f"{other}_team_elo"), errors="coerce")) / 400.0
                          if f"{side}_team_elo" in m else np.nan),
            }))
    tm = pd.concat(rows, ignore_index=True)
    tm = tm.sort_values(["date", "match_id", "is_home"]).reset_index(drop=True)
    order = {n: i for i, n in enumerate(seasons)}
    tm["season_idx"] = tm["season"].map(order)
    # per-team running index of matches played (for match-based recency of players)
    tm["team_match_idx"] = tm.groupby("team").cumcount()
    tm["season_match_no"] = tm.groupby(["team", "season"]).cumcount() + 1
    return tm


@dataclass
class StatFit:
    mu: float
    h: float
    a: dict[int, float]
    d: dict[int, float]
    prior_a: float
    prior_d: float


@dataclass
class TeamModel:
    asof: pd.Timestamp
    fits: dict[str, StatFit] = field(default_factory=dict)
    goal_scale: float = 1.0      # recent league goals per xG (xG slightly over-states goals)

    def expect(self, stat: str, team, opp, is_home) -> np.ndarray:
        f = self.fits[stat]
        team = np.asarray(team); opp = np.asarray(opp); is_home = np.asarray(is_home)
        a = np.array([f.a.get(int(t), f.prior_a) for t in np.atleast_1d(team)])
        d = np.array([f.d.get(int(o), f.prior_d) for o in np.atleast_1d(opp)])
        return np.exp(f.mu + f.h * np.atleast_1d(is_home) + a + d)

    def table(self, stat: str) -> pd.DataFrame:
        f = self.fits[stat]
        return pd.DataFrame({"team": list(f.a), "attack": [f.a[t] for t in f.a], "defence": [f.d[t] for t in f.a]})


def _fit_stat(df: pd.DataFrame, stat: str, params: Params, team_ids: list[int], prior: tuple[float, float]) -> StatFit:
    y = df[stat].to_numpy(float)
    w = df["w"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y, w = y[ok], w[ok]
    t_i = df["team"].to_numpy()[ok]; o_i = df["opp"].to_numpy()[ok]; home = df["is_home"].to_numpy(float)[ok]
    idx = {t: i for i, t in enumerate(team_ids)}
    ti = np.array([idx[t] for t in t_i]); oi = np.array([idx[o] for o in o_i])
    n = len(team_ids)
    ybar = float((w * y).sum() / w.sum())
    lam = params.shrink.team_ridge * ybar          # ridge in pseudo-observations at the league mean
    # which teams have real data in the window? the rest are pulled toward the promoted prior
    wsum = np.bincount(ti, weights=w, minlength=n) + np.bincount(oi, weights=w, minlength=n)
    no_data = wsum < 0.5 * np.median(wsum[wsum > 0]) * 0.1
    pa = np.where(no_data, prior[0], 0.0); pd_ = np.where(no_data, prior[1], 0.0)

    def f(x):
        mu, h = x[0], x[1]
        a, d = x[2:2 + n], x[2 + n:]
        eta = mu + h * home + a[ti] + d[oi]
        mu_hat = np.exp(eta)
        nll = -(w * (y * eta - mu_hat)).sum() + 0.5 * lam * (((a - pa) ** 2).sum() + ((d - pd_) ** 2).sum())
        r = w * (mu_hat - y)
        g = np.empty_like(x)
        g[0] = r.sum(); g[1] = (r * home).sum()
        g[2:2 + n] = np.bincount(ti, weights=r, minlength=n) + lam * (a - pa)
        g[2 + n:] = np.bincount(oi, weights=r, minlength=n) + lam * (d - pd_)
        return nll, g

    x0 = np.zeros(2 + 2 * n); x0[0] = np.log(max(ybar, 1e-6)); x0[2:2 + n] = pa; x0[2 + n:] = pd_
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    x = res.x
    return StatFit(float(x[0]), float(x[1]), {t: float(x[2 + idx[t]]) for t in team_ids},
                   {t: float(x[2 + n + idx[t]]) for t in team_ids}, prior[0], prior[1])


def fit_team_model(tm: pd.DataFrame, asof: pd.Timestamp, params: Params, current_season_idx: int,
                   all_teams: list[int], promoted_prior: dict | None = None,
                   stats: list[str] | None = None) -> TeamModel:
    """Fit all team statistics using only matches strictly before `asof`."""
    d = tm[tm["date"] < asof].copy()
    days = (asof - d["date"]).dt.total_seconds() / 86400.0
    hl_days = params.hl.team * 7.0   # config gives the half-life in weekly team-matches
    seasons_back = (current_season_idx - d["season_idx"]).clip(lower=0)
    d["w"] = 0.5 ** (days / hl_days) * (params.shrink.prior_season_weight ** seasons_back)
    prior = promoted_prior or DEFAULT_PROMOTED_PRIOR
    teams = sorted(set(all_teams) | set(d["team"]) | set(d["opp"]))
    out = TeamModel(asof)
    ok = d["goals_for"].notna() & d["xg"].notna()
    out.goal_scale = float(np.clip((d.loc[ok, "w"] * d.loc[ok, "goals_for"]).sum() /
                                   max((d.loc[ok, "w"] * d.loc[ok, "xg"]).sum(), 1e-9), 0.85, 1.15))
    for s in (stats or TEAM_STATS):
        out.fits[s] = _fit_stat(d, s, params, teams, prior.get(s, (0.0, 0.0)))
    return out
