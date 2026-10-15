"""Learned minutes model.

For a player who is available, what is the chance he plays 60+ minutes, 1-59 minutes, or not at all?
Features are exponentially weighted averages of his own recent starts/appearances (three memory
lengths), his last two matches, tenure at his club, early-season flag and position.  A multinomial
logistic regression maps them to probabilities and is refit walk-forward (only past data), so the
conversion from "recent usage" to "probability" is learned, not guessed.  Probabilities for a team's
players are then rescaled so that they add up to the number of players who actually play in a match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .fxfeat import rival_features

HLS = (2.0, 5.0, 12.0)           # memory lengths in the player's own available matches
POS = ["GKP", "DEF", "MID", "FWD"]
FEATS = (["e60_h2", "e60_h5", "e60_h12", "ep_h5", "em_h5", "last1", "last2", "conf", "e60_h5_c", "e60_h12_c",
          "es_h2", "es_h5", "es_h12", "esub_h5", "lasts1", "lasts2", "es_h5_c", "es_h12_c",
          "rest_b", "rest_a", "n14", "cong", "es5_cong", "rv_sum", "rv_max", "rv_gap",
          "inel_h4", "gap_log", "es5_inel", "retn", "lastmin", "retn_gap", "tenure", "early", "n_log"] + [f"pos_{p}" for p in POS])


def eligible(opp: pd.DataFrame) -> pd.Series:
    """Player was available to be picked (status a, or d with chance >=75). Unknown status (2024-25) counts."""
    s = opp["status"]
    ok = (s == "a") | ((s == "d") & (opp["chance"].fillna(0) >= 75)) | s.isna()
    return ok


def add_state(opp: pd.DataFrame) -> pd.DataFrame:
    """Add per-row EWMA state (`*_after`: includes that row; features use the previous row's state)."""
    o = opp.copy()
    o["_elig"] = eligible(o)
    o["_x60"] = (o["minutes"] >= 60).astype(float)
    o["_xp"] = (o["minutes"] > 0).astype(float)
    o["_xs"] = o["started"].astype(float) * o["_xp"]
    o["_xsub"] = (1.0 - o["started"].astype(float)) * o["_xp"]
    o["_xm"] = o["minutes"] / 90.0
    o = o.sort_values(["player_code", "date", "match_id"]).reset_index(drop=True)
    e = o[o["_elig"]]
    g = e.groupby("player_code", sort=False)
    st = pd.DataFrame(index=e.index)
    for h in HLS:
        a = 1 - 0.5 ** (1 / h)
        st[f"e60_h{int(h)}"] = g["_x60"].transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
    a5 = 1 - 0.5 ** (1 / 5.0)
    for h in HLS:
        a = 1 - 0.5 ** (1 / h)
        st[f"es_h{int(h)}"] = g["_xs"].transform(lambda s, a=a: s.ewm(alpha=a, adjust=False).mean())
    a5s = 1 - 0.5 ** (1 / 5.0)
    st["esub_h5"] = g["_xsub"].transform(lambda s: s.ewm(alpha=a5s, adjust=False).mean())
    st["lasts1"] = e["_xs"]
    st["lasts2"] = g["_xs"].shift(1)
    st["ep_h5"] = g["_xp"].transform(lambda s: s.ewm(alpha=a5, adjust=False).mean())
    st["em_h5"] = g["_xm"].transform(lambda s: s.ewm(alpha=a5, adjust=False).mean())
    st["n"] = g.cumcount() + 1
    st["last1"] = e["_x60"]
    st["last2"] = g["_x60"].shift(1)
    # return from absence: share of his recent team matches he was unavailable for, and days since he last played
    ol = o.sort_values(["player_code", "date", "match_id"])
    inel = (~ol["_elig"]).astype(float)
    a4 = 1 - 0.5 ** (1 / 4.0)
    ew = inel.groupby(ol["player_code"]).transform(lambda s: s.ewm(alpha=a4, adjust=False).mean())
    o["inel_after"] = ew.reindex(o.index)
    o["inel_pre"] = ew.groupby(ol["player_code"]).shift(1).fillna(0.0).reindex(o.index)
    lp = ol["date"].where(ol["minutes"] > 0).groupby(ol["player_code"]).ffill()
    o["lastplay_after"] = lp.reindex(o.index)
    o["lastplay_pre"] = lp.groupby(ol["player_code"]).shift(1).reindex(o.index)
    # first matches back after a long break (21+ days without playing): appearances since, and the last appearance's length
    app = (ol["minutes"] > 0)
    brk = app & ((pd.to_datetime(ol["date"], utc=True) - pd.to_datetime(o["lastplay_pre"].reindex(ol.index), utc=True)).dt.days >= 21)
    seg = brk.groupby(ol["player_code"]).cumsum()
    rn = app.astype(float).groupby([ol["player_code"], seg]).cumsum()
    o["retn_after"] = rn.reindex(o.index)
    o["retn_pre"] = rn.groupby(ol["player_code"]).shift(1).fillna(6.0).reindex(o.index)
    lm = ol["minutes"].where(app).groupby(ol["player_code"]).ffill()
    o["lastmin_after"] = (lm / 90.0).reindex(o.index)
    o["lastmin_pre"] = (lm / 90.0).groupby(ol["player_code"]).shift(1).reindex(o.index)
    # tenure: consecutive eligible rows at the current team
    chg = (e["team"] != g["team"].shift(1)).astype(int)
    run = chg.groupby(e["player_code"]).cumsum()
    st["tenure"] = e.groupby([e["player_code"], run]).cumcount() + 1
    st["season_no"] = e["season_idx"]
    # state AFTER each eligible row, carried forward over non-eligible rows
    after = st.reindex(o.index).groupby(o["player_code"]).ffill()
    for c in after.columns:
        o[c + "_after"] = after[c]
    return o


def gap_feature(date: pd.Series, last_play: pd.Series) -> pd.Series:
    days = (pd.to_datetime(date, utc=True) - pd.to_datetime(last_play, utc=True)).dt.total_seconds() / 86400.0
    return np.log1p(days.clip(0, 120).fillna(90.0)) / np.log(121.0)


def features_from_state(state: pd.DataFrame, position: pd.Series, early: pd.Series, prior60: dict,
                        extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """state: frame with the *_after columns of the previous available match (NaN if none).
    extra: per-row fixture / squad context (rest_b, rest_a, n14, elo_d, is_home, rv_sum, rv_max, inel_h4, gap_log)."""
    f = pd.DataFrame(index=state.index)
    pos = position.reindex(state.index)
    p60 = pos.map(prior60).fillna(0.3)
    n = state["n_after"].fillna(0.0)
    f["conf"] = np.minimum(n, 12) / 12.0
    for h in HLS:
        k = f"e60_h{int(h)}"
        f[k] = state[k + "_after"].fillna(p60)
    f["ep_h5"] = state["ep_h5_after"].fillna(p60 + 0.15)
    for h in HLS:
        k = f"es_h{int(h)}"
        f[k] = state[k + "_after"].fillna(p60)
    f["esub_h5"] = state["esub_h5_after"].fillna(0.12)
    f["lasts1"] = state["lasts1_after"].fillna(p60)
    f["lasts2"] = state["lasts2_after"].fillna(p60)
    f["es_h5_c"] = f["conf"] * (f["es_h5"] - p60)
    f["es_h12_c"] = f["conf"] * (f["es_h12"] - p60)
    f["em_h5"] = state["em_h5_after"].fillna(p60)
    f["last1"] = state["last1_after"].fillna(p60)
    f["last2"] = state["last2_after"].fillna(p60)
    f["e60_h5_c"] = f["conf"] * (f["e60_h5"] - p60)
    f["e60_h12_c"] = f["conf"] * (f["e60_h12"] - p60)
    ex = extra if extra is not None else pd.DataFrame(index=state.index)
    dflt = {"rest_b": 6.0, "rest_a": 6.0, "n14": 2.0, "elo_d": 0.0, "is_home": 0.5, "rv_sum": 0.0, "rv_max": 0.0,
            "inel_h4": 0.0, "gap_log": 0.2, "retn": 1.0, "lastmin": 0.8}
    for k, v in dflt.items():
        f[k] = ex[k].reindex(state.index).fillna(v) if k in ex else v
    f["cong"] = (f["rest_a"] < 4).astype(float) + (f["rest_b"] < 4).astype(float) + (f["n14"] >= 4).astype(float)
    f["es5_cong"] = f["es_h5"] * f["cong"]
    f["rv_gap"] = f["es_h5"] - f["rv_max"]
    f["es5_inel"] = f["es_h5"] * f["inel_h4"]
    f["retn_gap"] = (1.0 - f["retn"]) * f["es_h12"]
    f["tenure"] = np.minimum(state["tenure_after"].fillna(0), 10) / 10.0
    f["early"] = early.reindex(state.index).fillna(0).astype(float)
    f["n_log"] = np.log1p(n)
    for p in POS:
        f[f"pos_{p}"] = (pos == p).astype(float)
    return f[FEATS]


def training_frame(o: pd.DataFrame, prior60: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Rows of `add_state` output -> (features from the PREVIOUS state, 3-class target)."""
    prev = o.groupby("player_code", sort=False)[[c for c in o.columns if c.endswith("_after")]].shift(1)
    # previous *eligible* state is carried by ffill already; shifting by one row keeps it strictly past
    early = (o["season_match_no"] <= 4).astype(float)
    ex = pd.DataFrame({"rest_b": o["rest_b"], "rest_a": o["rest_a"], "n14": o["n14"], "elo_d": o["elo_d"], "is_home": o["is_home"],
                       "retn": np.minimum(o["retn_pre"], 6.0) / 6.0, "lastmin": o["lastmin_pre"],
                       "inel_h4": o["inel_pre"], "gap_log": gap_feature(o["date"], o["lastplay_pre"])}, index=o.index)
    X0 = features_from_state(prev, o["position"], early, prior60, ex)
    rv = rival_features(X0["es_h5"], o[["match_id", "team", "position"]])
    ex["rv_sum"], ex["rv_max"] = rv["rv_sum"], rv["rv_max"]
    X = features_from_state(prev, o["position"], early, prior60, ex)
    y = np.where(o["minutes"] <= 0, 0, np.where(o["started"] >= 1, 2, 1))     # 0 unused, 1 sub, 2 start
    return X, pd.Series(y, index=o.index)


@dataclass
class MinutesModel:
    lr: LogisticRegression
    prior60: dict

    @classmethod
    def fit(cls, o: pd.DataFrame, asof: pd.Timestamp, C: float = 0.5) -> "MinutesModel":
        pri = o[o["_elig"] & (o["date"] < asof)].groupby("position")["_x60"].mean().to_dict()
        d = o[(o["date"] < asof) & o["status"].notna() & o["_elig"] & (o["start_known"].astype(bool) | (o["minutes"] <= 0))]
        X, y = training_frame(o.loc[d.index], pri)
        # a transfer-window mover's old history is less informative; the tenure feature handles it
        import os
        if os.environ.get("MINUTES_MODEL", "hgb") == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            lr = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, l2_regularization=2.0, min_samples_leaf=60, random_state=0)
        else:
            lr = LogisticRegression(C=C, max_iter=400)
        lr.fit(X.to_numpy(), y.to_numpy())
        return cls(lr, pri)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pr = self.lr.predict_proba(X.to_numpy())
        cls_ = list(self.lr.classes_)
        out = pd.DataFrame({"p_start": pr[:, cls_.index(2)], "p_sub": pr[:, cls_.index(1)]}, index=X.index)
        return out


def latest_state(o: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """State after each player's last eligible match strictly before `asof` (one row per player)."""
    a = o[o["date"] < asof]
    last = a.groupby("player_code", sort=False).tail(1).set_index("player_code")
    return last[[c for c in o.columns if c.endswith("_after")]]


NORM_STRENGTH = 0.75      # 0 = trust each player's own probabilities, 1 = force the team totals (backtested: 0.5)


def normalise_team(df: pd.DataFrame, target60: float, targetplay: float, key=("match_id", "team"), strength: float = NORM_STRENGTH) -> pd.DataFrame:
    """Rescale the (availability-weighted) probabilities so each team-fixture's total matches what
    actually happens in a match (~9.3 players play 60+, ~14 players play at all)."""
    d = df.copy()
    k = list(key)
    s60 = d.groupby(k)["p60"].transform("sum")
    f60 = np.clip(target60 / s60.where(s60 > 0, np.nan), 0.6, 1.8).fillna(1.0)
    d["p60"] = np.clip(d["p60"] * f60, 0, 0.985)
    sp = (d["p60"] + d["pS"]).groupby([d[c] for c in k]).transform("sum")
    fp = np.clip(targetplay / sp.where(sp > 0, np.nan), 0.6, 1.8).fillna(1.0)
    tot = np.clip((d["p60"] + d["pS"]) * fp, 0, 0.995)
    d["pS"] = np.clip(tot - d["p60"], 0, 1)
    d["p60"] = (1 - strength) * df["p60"] + strength * d["p60"]
    d["pS"] = (1 - strength) * df["pS"] + strength * d["pS"]
    return d
