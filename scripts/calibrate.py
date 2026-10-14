"""Component-level calibration from the last backtest (config/_last_backtest.pkl)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.config import DC_THRESHOLD  # noqa: E402

pd.set_option("display.width", 200)
res = pd.read_pickle(Path(__file__).resolve().parent.parent / "config" / "_last_backtest.pkl")
d = res[(res.n_fixtures == 1)].dropna(subset=["event_points"]).copy()
st = d[(d.minutes >= 60)].copy()          # condition on playing 60+ to isolate the rate model from minutes
print(f"rows (single fixture): {len(d)}; played 60+: {len(st)}")


def table(x, y, by, q=6, label=""):
    t = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    t["b"] = pd.qcut(t["x"].rank(method="first"), q, labels=False)
    g = t.groupby("b").agg(n=("x", "size"), pred=("x", "mean"), actual=("y", "mean"))
    print(f"-- {label}: overall pred {t.x.mean():.4f} actual {t.y.mean():.4f} (ratio {t.y.mean() / max(t.x.mean(), 1e-9):.3f})")
    print(g.round(4).T.to_string())


# points conditional on 60+, by position
print("\n== ep_cond60 (expected points if he plays 60+) vs actual points among 60+ players")
for p, g in st.groupby("position"):
    print(f"  {p}: n={len(g)} pred {g.ep_cond60.mean():.3f} actual {g.event_points.mean():.3f} MAE {np.abs(g.ep_cond60 - g.event_points).mean():.3f}")
table(st.ep_cond60, st.event_points, st.position, 8, "points|60+ all positions")

out = st[st.position.isin(["MID", "FWD", "DEF"])]
# note lam_g/lam_a are for the expected minutes; scale to a full match for 60+ players: use actual minutes / m_exp
sc = out.minutes / out.m_exp.where(out.m_exp > 5, np.nan)
print("\n== goals: lam_g (scaled to actual minutes) vs goals scored")
for p, g in out.groupby("position"):
    s_ = (g.minutes / g.m_exp.where(g.m_exp > 5, np.nan))
    table(g.lam_g * s_, g.goals_scored, None, 5, p)
print("\n== assists")
for p, g in out.groupby("position"):
    s_ = (g.minutes / g.m_exp.where(g.m_exp > 5, np.nan))
    table(g.lam_a * s_, g.assists, None, 5, p)

print("\n== clean sheets (60+ players): p_cs / p_play vs clean sheet awarded")
cs = st[st.position.isin(["GKP", "DEF", "MID"])]
for p, g in cs.groupby("position"):
    table(g.p_cs / g.p_play.clip(lower=0.2), g.clean_sheets, None, 5, p)

print("\n== goals conceded penalty events: e_gc/p_play vs floor(goals_conceded/2) (GKP, DEF)")
gc = st[st.position.isin(["GKP", "DEF"])]
for p, g in gc.groupby("position"):
    table(g.e_gc / g.p_play.clip(lower=0.2), np.floor(g.goals_conceded / 2), None, 5, p)

print("\n== GK saves: e_saves vs saves")
g = st[st.position == "GKP"]
table(g.e_saves * (g.minutes / g.m_exp.where(g.m_exp > 5, np.nan)), g.saves, None, 5, "GKP")

print("\n== defensive contribution: p_dc / p_play vs hit (count >= threshold)")
for p in ["DEF", "MID", "FWD"]:
    g = st[st.position == p]
    hit = (g.defensive_contribution >= DC_THRESHOLD[p]).astype(float)
    table(g.p_dc / g.p_play.clip(lower=0.2), hit, None, 5, p)

print("\n== bonus: pts_bonus vs actual bonus (60+ players)")
for p, g in st.groupby("position"):
    table(g.pts_bonus, g.bonus, None, 5, p)
