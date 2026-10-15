"""Player layer: opportunity table, recency-decayed shares of team output, finishing/creation
ratios, minutes profile.

A player's stat x in a match is modelled as   x = share_x * T_x * (minutes / 90)
where T_x is the *team's* output of that kind in the match (from the team model for the future,
observed for the past).  Shares are estimated by ratio-of-sums with exponentially decaying weights
(per-attribute half-lives, counted in the player's team matches) and shrunk toward the position
average.  Because the share is relative to what the team produced, it carries no memory of how weak
or strong the opposition was.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params
from .data import Season
from .penalties import PEN_XG, add_pen_columns

EVENT_COLS = ["yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "bonus"]
STAT_COLS = ["xg", "xa", "goals", "assists", "saves", "dc_actions", "cbit", "xgot", "total_shots", "shots_on_target"]


def build_opportunities(seasons: dict[str, Season], tm: pd.DataFrame) -> pd.DataFrame:
    """One row per (player registered to a team, finished PL match of that team)."""
    parts = []
    for si, (name, s) in enumerate(seasons.items()):
        t = tm[tm["season"] == name][["match_id", "gw", "team", "opp", "is_home", "date", "xg", "sot_against", "cbit",
                                      "team_match_idx", "season_match_no"]].rename(columns={"xg": "T_xg", "sot_against": "T_sot_ag",
                                                                                           "cbit": "T_cbit"})
        reg = s.team_hist.rename(columns={"team_code": "team"})
        o = t.merge(reg, on=["team", "gw"], how="inner")
        o["season"], o["season_idx"] = name, si
        pos = s.players.set_index("player_code")["position"]
        o["position"] = o["player_code"].map(pos)

        pm = add_pen_columns(s.pms)
        PEN = ["pen_sc", "pen_ms", "pen_taken"]
        p = pm[["player_code", "match_id", "minutes_played"] + [c for c in STAT_COLS if c in pm] + PEN].copy()
        o = o.merge(p, on=["player_code", "match_id"], how="left")
        o["minutes"] = o["minutes_played"].fillna(0.0)
        for c in STAT_COLS + PEN:
            if c in o:
                o[c] = o[c].fillna(0.0)
        # players who appear in a match but were not registered to the team that GW are kept via pms
        extra = pm[["player_code", "match_id", "gw", "team_code", "opp_code", "is_home", "kickoff_time",
                    "minutes_played"] + [c for c in STAT_COLS if c in pm] + PEN].rename(
            columns={"team_code": "team", "opp_code": "opp", "kickoff_time": "date"})
        key = set(zip(o["player_code"], o["match_id"]))
        miss = extra[[(a, b) not in key for a, b in zip(extra["player_code"], extra["match_id"])]].copy()
        if len(miss):
            tt = t[["match_id", "team", "T_xg", "T_sot_ag", "T_cbit", "team_match_idx", "season_match_no"]]
            miss = miss.merge(tt, on=["match_id", "team"], how="left")
            miss["season"], miss["season_idx"] = name, si
            miss["position"] = miss["player_code"].map(pos)
            miss["minutes"] = miss["minutes_played"].fillna(0.0)
            o = pd.concat([o, miss], ignore_index=True)

        if not s.pgw.empty:
            g = s.pgw[["player_code", "gw", "defensive_contribution"] + [c for c in EVENT_COLS if c in s.pgw]].copy()
            g = g.rename(columns={"defensive_contribution": "dc_fpl"})
            o = o.merge(g.drop_duplicates(["player_code", "gw"], keep="last"), on=["player_code", "gw"], how="left")
            # A gameweek row's status/news/chance are recorded AFTER that gameweek (half of the players
            # flagged 'suspended' in GW g played in GW g). What was known before GW g is row g-1.
            st_prev = s.pgw[["player_code", "gw", "status", "chance_of_playing_next_round", "news"]].copy()
            st_prev = st_prev.rename(columns={"chance_of_playing_next_round": "chance"})
            st_prev["gw"] = st_prev["gw"] + 1
            o = o.merge(st_prev.drop_duplicates(["player_code", "gw"], keep="last"), on=["player_code", "gw"], how="left")
            # per-GW events are split across a double gameweek in proportion to minutes
            gm = o.groupby(["player_code", "gw"])["minutes"].transform("sum")
            frac = np.where(gm > 0, o["minutes"] / gm.replace(0, np.nan), 0.0)
            for c in EVENT_COLS:
                if c in o:
                    o[c] = o[c].fillna(0.0) * np.nan_to_num(frac)
            o["has_pgw"] = True
            # FPL's own defensive-contribution count is what awards the points: prefer it whenever the
            # player had a single match that gameweek (a double gameweek count cannot be split)
            nm = (o["minutes"] > 0).groupby([o["player_code"], o["gw"]]).transform("sum")
            use = (o["minutes"] > 0) & (nm == 1) & o["dc_fpl"].notna() & o["position"].isin(["DEF", "MID", "FWD"])
            o.loc[use, "dc_actions"] = o.loc[use, "dc_fpl"]
            o = o.drop(columns=["dc_fpl"])
        else:
            o["status"], o["chance"], o["news"], o["has_pgw"] = np.nan, np.nan, np.nan, False
            for c in EVENT_COLS:
                o[c] = np.nan
        # open-play versions: penalties are modelled separately (penalties.py)
        o["xg_open"] = (o["xg"] - PEN_XG * o["pen_taken"]).clip(lower=0.0)
        o["goals_open"] = (o["goals"] - o["pen_sc"]).clip(lower=0.0)
        parts.append(o)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["minutes_played"], errors="ignore")
    out = out[out["position"].notna() & out["date"].notna()]
    return out.sort_values(["date", "match_id", "player_code"]).reset_index(drop=True)


@dataclass
class Profiles:
    table: pd.DataFrame          # one row per player_code
    pos_prior: pd.DataFrame      # position-level priors actually used


def _wsum(df: pd.DataFrame, w: np.ndarray, num: str | np.ndarray, den: str | np.ndarray, by="player_code"):
    n = df[num] if isinstance(num, str) else num
    d = df[den] if isinstance(den, str) else den
    g = pd.DataFrame({by: df[by].to_numpy(), "n": np.asarray(n) * w, "d": np.asarray(d) * w})
    return g.groupby(by).sum()


def player_profiles(opp: pd.DataFrame, asof: pd.Timestamp, params: Params, team_idx_asof: dict[int, int],
                    season_idx_now: int, positions: pd.Series) -> Profiles:
    """All per-player quantities the predictor needs, using only matches strictly before `asof`."""
    a = opp[opp["date"] < asof].copy()
    age = a["team"].map(team_idx_asof).fillna(0).to_numpy() - a["team_match_idx"].to_numpy() - 1
    age = np.clip(age, 0, None)
    back = np.clip(season_idx_now - a["season_idx"].to_numpy(), 0, None)
    sf = params.shrink.prior_season_weight ** back

    def W(hl: float) -> np.ndarray:
        return (0.5 ** (age / hl)) * sf

    played = (a["minutes"] > 0).to_numpy()
    m90 = a["minutes"].to_numpy() / 90.0
    k = params.shrink.rate_matches
    prof = pd.DataFrame(index=pd.Index(sorted(set(positions.index)), name="player_code"))
    prof["position"] = positions
    pri = {}

    def share(name, num_col, T_col, hl, mean_T):
        w = W(hl) * played
        expo = a[T_col].fillna(a[T_col].mean()).to_numpy() * m90
        g = _wsum(a, w, num_col, expo)
        pos = a["position"].to_numpy()
        gp = pd.DataFrame({"pos": pos, "n": a[num_col].to_numpy() * w, "d": expo * w}).groupby("pos").sum()
        prior = (gp["n"] / gp["d"]).rename("prior")
        pri[name] = prior
        merged = prof[["position"]].join(g, how="left").fillna({"n": 0.0, "d": 0.0})
        pp = merged["position"].map(prior).fillna(0.0)
        kk = k * mean_T
        prof[name] = (merged["n"] + kk * pp) / (merged["d"] + kk)
        prof[name + "_expo"] = merged["d"]

    mean_Txg = float(a["T_xg"].mean()); mean_Tc = float(a["T_cbit"].mean()); mean_Ts = float(a["T_sot_ag"].mean())
    share("xg_share", "xg_open", "T_xg", params.hl.xg, mean_Txg)
    share("xa_share", "xa", "T_xg", params.hl.xa, mean_Txg)
    share("dc_share", "dc_actions", "T_cbit", params.hl.cbit, mean_Tc)
    # shots on target faced -> saves (goalkeepers only)
    gk = a["position"] == "GKP"
    w = W(params.hl.saves) * played * gk.to_numpy()
    expo = a["T_sot_ag"].fillna(mean_Ts).to_numpy() * m90
    g = _wsum(a, w, "saves", expo)
    prior_save = float(g["n"].sum() / max(g["d"].sum(), 1e-9)) if len(g) else 0.7
    kk = k * mean_Ts
    mg = prof[["position"]].join(g, how="left").fillna({"n": 0.0, "d": 0.0})
    prof["save_ratio"] = (mg["n"] + kk * prior_save) / (mg["d"] + kk)
    pri["save_ratio"] = pd.Series({"GKP": prior_save})

    # finishing and creation: actual vs expected, shrunk to 1
    for name, act, exp_, hl, c in (("finish", "goals_open", "xg_open", params.hl.goals, params.shrink.finishing_xg),
                                   ("create", "assists", "xa", params.hl.assists, params.shrink.creation_xa)):
        w = W(hl) * played
        g = pd.DataFrame({"player_code": a["player_code"].to_numpy(), "A": a[act].to_numpy() * w,
                          "X": a[exp_].to_numpy() * w}).groupby("player_code").sum()
        gp = pd.DataFrame({"pos": a["position"].to_numpy(), "A": a[act].to_numpy() * w,
                           "X": a[exp_].to_numpy() * w}).groupby("pos").sum()
        r_pos = (gp["A"] / gp["X"].where(gp["X"] > 1.0, np.nan)).clip(0.4, 3.0)   # league ratio by position
        pri[name + "_ratio"] = r_pos
        mg = prof[["position"]].join(g, how="left").fillna(0.0)
        rp = mg["position"].map(r_pos).fillna(1.0)
        prof[name + "_ratio"] = (mg["A"] + c * rp) / (mg["X"] + c)

    # yellow cards / red cards per 90 (needs per-GW table => 2025-26 onward)
    h = a[a["has_pgw"].fillna(False).astype(bool)]
    hw = (0.5 ** (np.clip(h["team"].map(team_idx_asof).fillna(0).to_numpy() - h["team_match_idx"].to_numpy() - 1, 0, None)
                  / params.hl.cards)) * params.shrink.prior_season_weight ** np.clip(season_idx_now - h["season_idx"].to_numpy(), 0, None)
    hp = (h["minutes"] > 0).to_numpy()
    for ev, name in (("yellow_cards", "yc90"), ("red_cards", "rc90"), ("own_goals", "og90"),
                     ("penalties_missed", "pm90"), ("penalties_saved", "ps90")):
        gnum = pd.DataFrame({"player_code": h["player_code"].to_numpy(), "n": h[ev].to_numpy() * hw * hp,
                             "d": h["minutes"].to_numpy() / 90.0 * hw * hp}).groupby("player_code").sum()
        gp = pd.DataFrame({"pos": h["position"].to_numpy(), "n": h[ev].to_numpy() * hw * hp,
                           "d": h["minutes"].to_numpy() / 90.0 * hw * hp}).groupby("pos").sum()
        prior = (gp["n"] / gp["d"]).rename("prior")
        mg = prof[["position"]].join(gnum, how="left").fillna(0.0)
        kk = 8.0 if ev in ("yellow_cards",) else 40.0   # pseudo 90s: red cards / OGs / pens are too rare to personalise
        pp = mg["position"].map(prior).fillna(0.0)
        prof[name] = (mg["n"] + kk * pp) / (mg["d"] + kk)
        pri[name] = prior

    # minutes profile: only matches where the player was available to be picked (status 'a', or 'd' >=75)
    has_status = a["status"].notna()
    avail = ((a["status"] == "a") | ((a["status"] == "d") & (a["chance"].fillna(0) >= 75))) & has_status
    wm = W(params.hl.minutes) * avail.to_numpy()
    mm = a["minutes"].to_numpy()
    cat60 = (mm >= 60).astype(float); catS = ((mm > 0) & (mm < 60)).astype(float)
    g = pd.DataFrame({"player_code": a["player_code"].to_numpy(), "w": wm, "p60": cat60 * wm, "pS": catS * wm,
                      "m60": mm * cat60 * wm, "mS": mm * catS * wm}).groupby("player_code").sum()
    gp = pd.DataFrame({"pos": a["position"].to_numpy(), "w": wm, "p60": cat60 * wm, "pS": catS * wm,
                       "m60": mm * cat60 * wm, "mS": mm * catS * wm}).groupby("pos").sum()
    pos_p60 = (gp["p60"] / gp["w"]); pos_pS = (gp["pS"] / gp["w"])
    pos_m60 = (gp["m60"] / gp["p60"]); pos_mS = (gp["mS"] / gp["pS"])
    mg = prof[["position"]].join(g, how="left").fillna(0.0)
    kmin = 2.0
    pp60 = mg["position"].map(pos_p60).fillna(0.25) * 0.0 + 0.25      # a stranger is a squad player until shown otherwise
    ppS = mg["position"].map(pos_pS).fillna(0.15)
    prof["p60"] = (mg["p60"] + kmin * pp60) / (mg["w"] + kmin)
    prof["pS"] = (mg["pS"] + kmin * ppS) / (mg["w"] + kmin)
    prof["m60"] = ((mg["m60"] + 3 * mg["position"].map(pos_m60).fillna(88)) / (mg["p60"] + 3)).clip(60, 95)
    prof["mS"] = ((mg["mS"] + 3 * mg["position"].map(pos_mS).fillna(22)) / (mg["pS"] + 3)).clip(5, 55)
    prof["n_eff_minutes"] = mg["w"]
    # appearance of the player in the minutes model at all (history length)
    prof["has_history"] = prof["n_eff_minutes"] > 0
    pri["p60"], pri["pS"] = pos_p60, pos_pS
    return Profiles(prof, pd.DataFrame(pri))
