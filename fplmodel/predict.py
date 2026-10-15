"""Expected FPL points per player per fixture, assembled component by component.

Every scoring event is turned into an expectation using the right distribution, because several
FPL rules are non-linear in the underlying count:
    goals, assists           Poisson (rates from share x team xG x minutes)
    clean sheet              exp(-expected goals against over the minutes played), needs 60+ min
    goals conceded           E[floor(G/2)] with G ~ Poisson            (GKP/DEF only)
    saves                    E[floor(S/3)] with S ~ Poisson            (GKP only)
    defensive contribution   P(count >= threshold), count ~ NegBin    (DEF 10, MID/FWD 12)
    appearance               P(60+) * 2 + P(1-59) * 1
    bonus                    regression on the components above (fitted in the backtest)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import CONFIG_DIR, DC_THRESHOLD, Params
from .bonus import simulate_bonus
from .minutes import normalise_team
from .scoring import Rules
from .teams import TeamModel

POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
_N = np.arange(0, 31)


def e_floor_div(lam: np.ndarray, d: int) -> np.ndarray:
    """E[floor(N/d)] for N ~ Poisson(lam)."""
    lam = np.clip(np.asarray(lam, float), 1e-9, 40.0)
    pm = stats.poisson.pmf(_N[None, :], lam[:, None])
    return pm @ (_N // d).astype(float)


def p_dc_hit(mu: np.ndarray, thr: np.ndarray, alpha: float) -> np.ndarray:
    """P(X >= thr) for X ~ NegBin with mean mu and var = mu + alpha mu^2."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    r = 1.0 / alpha
    p = r / (r + mu)
    out = stats.nbinom.sf(np.asarray(thr) - 1, r, p)
    return np.where(np.isfinite(thr), out, 0.0)


def fixture_context(fix: pd.DataFrame, tmodel: TeamModel) -> pd.DataFrame:
    """Expected team-level quantities for each (team, fixture). `fix` has team, opp, is_home."""
    f = fix.copy()
    f["lam_for"] = tmodel.expect("xg", f["team"], f["opp"], f["is_home"])
    f["lam_ag"] = tmodel.expect("xg", f["opp"], f["team"], 1 - f["is_home"]) * tmodel.goal_scale
    f["sot_ag"] = tmodel.expect("sot", f["opp"], f["team"], 1 - f["is_home"])
    f["t_cbit"] = tmodel.expect("cbit", f["team"], f["opp"], f["is_home"])
    return f


@dataclass
class BonusModel:
    coef: dict   # position -> {feature: weight, "const": c}

    @classmethod
    def load(cls) -> "BonusModel":
        f = CONFIG_DIR / "bonus_model.json"
        if f.exists():
            return cls(json.loads(f.read_text()))
        # conservative defaults until the backtest fits them
        d = {p: {"const": 0.0, "lam_g": 0.9, "lam_a": 0.55, "p_cs": 0.25 if p in ("GKP", "DEF") else 0.1,
                 "p60": 0.12, "e_saves": 0.05} for p in POS_ORDER}
        return cls(d)

    def predict(self, comp: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(comp))
        for p in POS_ORDER:
            m = (comp["position"] == p).to_numpy()
            if not m.any():
                continue
            c = self.coef[p]
            v = np.full(m.sum(), c.get("const", 0.0))
            for k, w in c.items():
                if k != "const" and k in comp:
                    v = v + w * comp.loc[m, k].to_numpy()
            out[m] = np.clip(v, 0, 3)
        return out


def expected_points(ctx: pd.DataFrame, prof: pd.DataFrame, avail: pd.DataFrame, rules: Rules, params: Params,
                    bonus: BonusModel | None = None, norm: tuple | None = None,
                    mins_fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """ctx: per (team, fixture) from fixture_context; prof: per player (needs team_code and position);
    avail: per (player_code, gw_rel) -> factor.  Returns one row per player x fixture with components."""
    bonus = bonus or BonusModel.load()
    d = prof.reset_index().merge(ctx, left_on="team_code", right_on="team", how="inner")
    d = d.merge(avail, on=["player_code", "gw"], how="left")
    d["avail"] = d["avail"].fillna(1.0)
    if mins_fx is not None and "p_start" in d:
        # fixture-specific start / sub probabilities (rest, rival strength, ...) replace the player-level ones
        d = d.merge(mins_fx, on=["player_code", "match_id"], how="left")
        ok = d["p_start_fx"].notna() & ~d["mins_fixed"].fillna(False).astype(bool)
        q0 = d["q60s"].clip(0.05, 1.0)
        d["p_start"] = np.where(ok, d["p_start_fx"], d["p_start"])
        d["p_sub"] = np.where(ok, d["p_sub_fx"], d["p_sub"])
        d["p60"] = np.where(ok, d["p_start"] * q0, d["p60"])
        d["pS"] = np.where(ok, d["p_start"] * (1 - q0) + d["p_sub"], d["pS"])
    pos = d["position"]
    p60 = d["avail"] * d["p60"]
    pS = d["avail"] * d["pS"]
    pS = np.minimum(pS, 1 - p60)
    if norm is not None:
        tmp = normalise_team(pd.DataFrame({"match_id": d["match_id"], "team": d["team"], "p60": p60, "pS": pS}), *norm)
        p60, pS = tmp["p60"], tmp["pS"]
    m60 = d["m60"].to_numpy()
    if "p_start" in d:
        # a starter reaches 60 with probability q; the rest of the appearances are early hooks and substitutes
        q = d["q60s"].clip(0.05, 1.0).to_numpy()
        pst = np.clip(np.asarray(p60) / q, 0, 1)
        phook = pst * (1 - q)
        psub = np.clip(np.asarray(pS) - phook, 0, 1)
        m_exp = p60 * m60 + phook * d["m_hook"].to_numpy() + psub * d["m_sub"].to_numpy()
        mS = np.where(np.asarray(pS) > 1e-6, (m_exp - p60 * m60) / np.maximum(np.asarray(pS), 1e-6), d["m_sub"].to_numpy())
        d["mS"] = mS
        d["p_start_eff"], d["p_sub_eff"] = pst, psub
        d["m_from_start"] = p60 * m60 + phook * d["m_hook"].to_numpy()      # expected minutes that come from starting
        d["m_from_sub"] = psub * d["m_sub"].to_numpy()                      # ... and from coming off the bench
    else:
        mS = d["mS"].to_numpy()
        m_exp = p60 * m60 + pS * mS
    d["p_play"] = p60 + pS
    # penalties: designated takers weighted by how much of the match they are expected to be on the pitch
    if "pen_w" in d:
        wq = d["pen_w"].fillna(0.0) * (m_exp / 90.0)
        tot = wq.groupby([d["match_id"], d["team"]]).transform("sum")
        d["pen_lam_min"] = np.where(tot > 1e-9, d["pen_rate"] * d["pen_w"].fillna(0.0) / np.where(tot > 1e-9, tot, 1.0) / 90.0, 0.0)
    d["p60_eff"] = p60
    d["m_exp"] = m_exp

    pt = lambda k: pos.map(lambda p: rules.pts(k, p)).to_numpy(float)  # noqa: E731
    d["_p60"], d["_pS"] = p60, pS
    # bonus points: simulate each match's BPS race (needs the per-90 rates the components use)
    pen90 = d["pen_lam_min"].to_numpy() * 90.0 * (float(d["pen_conv"].iloc[0]) if "pen_conv" in d and len(d) else 0.8) if "pen_lam_min" in d else 0.0
    is_gk = (pos == "GKP").to_numpy()
    rates = {"g": (d["finish_ratio"] * d["xg_share"] * d["lam_for"]).to_numpy() + pen90,
             "a": (d["create_ratio"] * d["xa_share"] * d["lam_for"]).to_numpy(),
             "sv": np.where(is_gk, (d["save_ratio"] * d["sot_ag"]).to_numpy(), 0.0),
             "yc": d["yc90"].to_numpy(), "rc": d["rc90"].to_numpy(), "og": d["og90"].to_numpy(),
             "pm": np.zeros(len(d)), "ps": d["ps90"].to_numpy(), "dc": (d["dc_share"] * d["t_cbit"]).to_numpy(),
             "ag": d["lam_ag"].to_numpy()}
    sim = simulate_bonus(d.reset_index(drop=True), rates, p60.to_numpy(), pS.to_numpy())
    if sim is not None:
        d["bonus_sim"], d["bonus_sim60"] = sim
    comp = _components(d, pt, rules, params, bonus, p60, pS)
    cond = _components(d, pt, rules, params, bonus, pd.Series(1.0, index=d.index), pd.Series(0.0, index=d.index))
    d["ep_cond60"] = cond.drop(columns=[c for c in cond.columns if c.startswith("_")]).sum(axis=1)
    keep = [c for c in comp.columns if not c.startswith("_")]
    d = pd.concat([d, comp[keep].add_prefix("pts_"), comp[[c for c in comp.columns if c.startswith("_")]].rename(columns=lambda c: c[1:])], axis=1)
    d = d.loc[:, ~d.columns.duplicated(keep="last")]
    pcols = ["pts_" + c for c in keep]
    d["ep"] = d[pcols].sum(axis=1)

    var = (pt("goals_scored") ** 2 * d["lam_g"] + rules.pts("assists", "MID") ** 2 * d["lam_a"] +
           pt("clean_sheets") ** 2 * d["p_cs"] * (1 - d["p_cs"]) + 4 * d["p_dc"] * (1 - d["p_dc"]) + 0.8 * (d["pts_bonus"] > 0) +
           (d["_p60"] * (1 - d["_p60"])) * 1.0)
    d["ep_sd"] = np.sqrt(var + 1e-9)
    return d


def _components(d: pd.DataFrame, pt, rules: Rules, params: Params, bonus: BonusModel, p60: pd.Series, pS: pd.Series) -> pd.DataFrame:
    """Points by scoring component for given 60+/short-appearance probabilities. Columns starting
    with '_' are diagnostics (rates) rather than points."""
    pos = d["position"]
    m60, mS = d["m60"].to_numpy(), d["mS"].to_numpy()
    m_exp = (p60 * m60 + pS * mS).to_numpy()
    lam_g = (d["finish_ratio"] * d["xg_share"] * d["lam_for"]).to_numpy() * m_exp / 90.0
    lam_a = (d["create_ratio"] * d["xa_share"] * d["lam_for"]).to_numpy() * m_exp / 90.0
    if "pen_lam_min" in d:
        conv = float(d["pen_conv"].iloc[0]) if "pen_conv" in d and len(d) else 0.8
        lam_pen = d["pen_lam_min"].to_numpy() * m_exp          # penalties he is expected to take
    else:
        conv, lam_pen = 0.8, np.zeros(len(d))
    lam_g = lam_g + lam_pen * conv                          # open play + converted penalties
    p_cs = (p60 * np.exp(-d["lam_ag"] * m60 / 90.0)).to_numpy()
    e_gc = (p60.to_numpy() * e_floor_div(d["lam_ag"].to_numpy() * m60 / 90.0, 2) +
            pS.to_numpy() * e_floor_div(d["lam_ag"].to_numpy() * mS / 90.0, 2))
    is_gk = (pos == "GKP").to_numpy()
    mu_s = (d["save_ratio"] * d["sot_ag"]).to_numpy()
    e_sv = np.where(is_gk, p60.to_numpy() * e_floor_div(mu_s * m60 / 90.0, 3) + pS.to_numpy() * e_floor_div(mu_s * mS / 90.0, 3), 0.0)
    e_saves = np.where(is_gk, mu_s * m_exp / 90.0, 0.0)
    thr = pos.map(lambda p: DC_THRESHOLD.get(p) or np.inf).to_numpy(float)
    mu60 = (d["dc_share"] * d["t_cbit"]).to_numpy() * m60 / 90.0
    muS = (d["dc_share"] * d["t_cbit"]).to_numpy() * mS / 90.0
    p_dc = p60.to_numpy() * p_dc_hit(mu60, thr, params.nb_dispersion_dc) + pS.to_numpy() * p_dc_hit(muS, thr, params.nb_dispersion_dc)

    c = pd.DataFrame(index=d.index)
    c["appearance"] = p60 * rules.long_play + pS * rules.short_play
    c["goals"] = lam_g * pt("goals_scored")
    c["assists"] = lam_a * rules.pts("assists", "MID")
    c["clean_sheet"] = p_cs * pt("clean_sheets")
    c["goals_conceded"] = e_gc * pt("goals_conceded")
    c["saves"] = e_sv * rules.pts("saves", "GKP")
    c["defensive"] = p_dc * pt("defensive_contribution")
    c["cards"] = (d["yc90"] * rules.pts("yellow_cards", "MID") + d["rc90"] * rules.pts("red_cards", "MID")) * m_exp / 90.0
    pen_miss = (lam_pen * (1.0 - conv) * rules.pts("penalties_missed", "MID")) if "pen_lam_min" in d else (d["pm90"] * rules.pts("penalties_missed", "MID") * m_exp / 90.0)
    c["other"] = (d["og90"] * rules.pts("own_goals", "MID") + d["ps90"] * rules.pts("penalties_saved", "GKP")) * m_exp / 90.0 + pen_miss
    c["_lam_pen"] = lam_pen
    use_sim = "bonus_sim" in d
    bon_in = pd.DataFrame({"position": pos.to_numpy(), "lam_g": lam_g, "lam_a": lam_a, "p_cs": p_cs, "e_saves": e_saves,
                           "p60": p60.to_numpy()}, index=d.index)
    if use_sim:
        c["bonus"] = (d["bonus_sim60"] if bool((p60.to_numpy() == 1).all()) else d["bonus_sim"]).to_numpy() * rules.pts("bonus", "MID")
    else:
        c["bonus"] = bonus.predict(bon_in) * rules.pts("bonus", "MID")
    c["_lam_g"], c["_lam_a"], c["_p_cs"], c["_p_dc"], c["_e_saves"], c["_m_exp"] = lam_g, lam_a, p_cs, p_dc, e_saves, m_exp
    c["_e_gc"], c["_p_play"] = e_gc, (p60 + pS).to_numpy()
    return c


def aggregate_gameweek(d: pd.DataFrame) -> pd.DataFrame:
    """Sum fixtures within a gameweek (double gameweeks add up; blank gameweeks have no row)."""
    cols = [c for c in d.columns if c.startswith("pts_")] + ["ep", "ep_cond60", "p_play", "p60_eff", "p_start_eff", "p_sub_eff", "m_from_start", "m_from_sub", "m_exp", "lam_g", "lam_a", "p_cs", "p_dc", "e_saves", "e_gc"]
    g = d.groupby(["player_code", "gw"], as_index=False).agg({**{c: "sum" for c in cols}, "ep_sd": lambda x: float(np.sqrt((x ** 2).sum())),
                                                             "match_id": "count"})
    return g.rename(columns={"match_id": "n_fixtures"})
