"""Bonus points from a match-level simulation of the Bonus Points System (BPS).

FPL gives 3/2/1 bonus points to the three highest BPS in a match (both teams together; ties share).
A player's bonus therefore depends on who else is in the match, which a per-player regression cannot see.

For each match we simulate `sims` matches:
  * every player is on the pitch for 60+ minutes, a short spell, or not at all (p60, pS from the minutes model)
  * goals, assists, saves and yellow cards are Poisson draws from the player's rates; each team concedes a
    Poisson number of goals shared by its players, so one clean sheet lifts a whole back line together
  * BPS = fitted position-specific weights x events + the stats we cannot forecast (passing, tackling,
    chances), entered through their expectation plus a fitted residual noise
  * the top three BPS are awarded 3/2/1 with FPL's tie rule (a tied place shares the points, the next is skipped)
Weights and noise are fitted by scripts/fit_bps.py from real BPS against real events (R2 0.87-0.95).
"""
from __future__ import annotations

import json
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import CONFIG_DIR

KEYS = ["const", "i60", "goals", "assists", "cs", "gc", "saves", "yc", "rc", "og", "pm", "ps", "dc", "xa", "xg", "sd"]


@lru_cache(maxsize=1)
def load_bps() -> dict | None:
    f = CONFIG_DIR / "bps_model.json"
    return json.loads(f.read_text()) if f.exists() else None


def award(bps: np.ndarray) -> np.ndarray:
    """3/2/1 bonus for the top three BPS across the last axis' players, FPL tie rule (tied players share
    the place, the next place is skipped). bps shape (..., n); players who did not play carry -inf/very low."""
    minrank = 1 + (bps[..., None, :] > bps[..., :, None]).sum(axis=-1)
    return np.where(minrank == 1, 3.0, np.where(minrank == 2, 2.0, np.where(minrank == 3, 1.0, 0.0)))


def simulate_bonus(d: pd.DataFrame, rates: dict[str, np.ndarray], p60: np.ndarray, pS: np.ndarray, sims: int = 1000,
                   seed: int = 20260101) -> tuple[np.ndarray, np.ndarray] | None:
    """Expected bonus points per row of `d` (one row per player x fixture), and the expectation given the
    player plays 60+ minutes.  `rates` holds per-90 rates keyed g, a, sv, yc, rc, og, pm, ps, dc and the
    team's expected goals against `ag`; arrays are aligned with d."""
    cfg = load_bps()
    if cfg is None:
        return None
    n_all = len(d)
    out, out60 = np.zeros(n_all), np.zeros(n_all)
    pos = d["position"].to_numpy()
    coef = np.array([[cfg[p][k] for k in KEYS] for p in pos])               # (rows, 16)
    m60, mS = d["m60"].to_numpy(float), d["mS"].to_numpy(float)
    team = d["team"].to_numpy()
    for mid, idx in d.groupby("match_id").indices.items():
        idx = np.asarray(idx)
        keep = idx[(p60[idx] + pS[idx]) > 0.01]
        if len(keep) < 3:
            continue
        rng = np.random.default_rng(seed + zlib.crc32(str(mid).encode()) % 1_000_003)
        n = len(keep)
        u = rng.random((sims, n))
        a60, aS = p60[keep][None, :], pS[keep][None, :]
        state = np.where(u < a60, 2, np.where(u < a60 + aS, 1, 0))
        frac = np.where(state == 2, m60[keep][None, :] / 90.0, np.where(state == 1, mS[keep][None, :] / 90.0, 0.0))
        c = coef[keep].T[:, None, :]                                         # (16, 1, n)
        R = {k: v[keep][None, :] for k, v in rates.items()}
        G = rng.poisson(R["g"] * frac)
        A = rng.poisson(R["a"] * frac)
        SV = rng.poisson(R["sv"] * frac)
        YC = rng.poisson(R["yc"] * frac)
        tl = team[keep]
        teams = list(dict.fromkeys(tl.tolist()))
        ti = np.array([teams.index(t) for t in tl])
        ag = np.array([R["ag"][0, np.where(ti == k)[0][0]] for k in range(len(teams))])
        Gt = rng.poisson(ag[None, :], size=(sims, len(teams)))
        Gc = Gt[:, ti]
        CS = ((state == 2) & (Gc == 0)).astype(float)
        GC = rng.binomial(Gc, np.clip(frac, 0, 1))
        K = dict(zip(KEYS, c))
        bps = (K["const"] + K["i60"] * (state == 2) + K["goals"] * G + K["assists"] * A + K["cs"] * CS + K["gc"] * GC
               + K["saves"] * SV + K["yc"] * YC
               + (K["rc"] * R["rc"] + K["og"] * R["og"] + K["pm"] * R["pm"] + K["ps"] * R["ps"] + K["dc"] * R["dc"]
                  + K["xa"] * R["a"] + K["xg"] * R["g"]) * frac
               + K["sd"] * rng.standard_normal((sims, n)))
        bps = np.rint(bps)
        bps[state == 0] = -1e6
        bonus = award(bps) * (state > 0)
        out[keep] = bonus.mean(axis=0)
        s2 = (state == 2)
        out60[keep] = (bonus * s2).sum(axis=0) / np.maximum(s2.sum(axis=0), 1)
        # a player who is almost never on the pitch has no 60+ sample: fall back to the unconditional figure
        out60[keep] = np.where(s2.sum(axis=0) >= 20, out60[keep], out[keep] / np.maximum(p60[keep], 0.05))
    return out, out60
