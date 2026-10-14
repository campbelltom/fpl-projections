"""Official FPL scoring. Rules are read from the live API (game_config.scoring) when the
network allows, falling back to the committed snapshot, so a rule change can never
silently leave the model stale.

Per-event structure (all by position unless noted):
  appearance      1 pt for 1-59 min, 2 pts for 60+            (long_play / short_play)
  goals_scored    GKP 10, DEF 6, MID 5, FWD 4
  assists         3
  clean_sheets    GKP/DEF 4, MID 1, FWD 0  (needs 60+ min and no goals conceded while on pitch)
  goals_conceded  GKP/DEF -1 per 2 conceded
  saves           1 per 3 saves (GKP)
  penalties       saved +5, missed -2
  cards           yellow -1, red -3 ; own goal -2
  bonus           1-3
  defensive_contribution  +2 once per match when the CBIT (DEF) / CBIRT (MID,FWD) threshold is hit
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_DIR, DC_THRESHOLD

log = logging.getLogger("fplmodel.scoring")
API = "https://fantasy.premierleague.com/api/bootstrap-static/"
SNAPSHOT = CONFIG_DIR / "scoring_snapshot.json"


@dataclass
class Rules:
    raw: dict
    source: str

    def pts(self, key: str, pos: str) -> float:
        v = self.raw[key]
        return float(v[pos]) if isinstance(v, dict) else float(v)

    @property
    def long_play(self) -> float:
        return float(self.raw["long_play"])

    @property
    def short_play(self) -> float:
        return float(self.raw["short_play"])


def load_rules(live: bool = True, timeout: float = 15.0) -> Rules:
    if live:
        try:
            req = urllib.request.Request(API, headers={"User-Agent": "fpl-projections/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            raw = data["game_config"]["scoring"]
            _check_shape(raw)
            return Rules(raw, "live-api")
        except Exception as e:  # network blocked, API down, schema change
            log.warning("live scoring fetch failed (%s); using snapshot", e)
    raw = json.loads(SNAPSHOT.read_text())
    _check_shape(raw)
    return Rules(raw, "snapshot")


def _check_shape(raw: dict) -> None:
    need = ["long_play", "short_play", "goals_scored", "assists", "clean_sheets",
            "goals_conceded", "saves", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards", "own_goals", "bonus", "defensive_contribution"]
    miss = [k for k in need if k not in raw]
    if miss:
        raise ValueError(f"scoring table missing keys: {miss}")


def dc_hit(pos: pd.Series, dc_actions: pd.Series) -> pd.Series:
    """Boolean: did the DC action count reach the position threshold."""
    thr = pos.map(lambda p: DC_THRESHOLD.get(p) or np.inf)
    return dc_actions >= thr


def actual_points(df: pd.DataFrame, rules: Rules, bonus: bool = True, dc: bool = True) -> pd.Series:
    """Reconstruct FPL points from realised events (one row = one player-match or single-match GW).

    Needs columns: position, minutes, goals_scored, assists, clean_sheets, goals_conceded, saves,
    penalties_saved, penalties_missed, yellow_cards, red_cards, own_goals, bonus,
    defensive_contribution (the raw action count).
    """
    pos = df["position"]
    mins = df["minutes"].fillna(0)
    pt = lambda k: pos.map(lambda p: rules.pts(k, p))  # noqa: E731
    out = np.where(mins >= 60, rules.long_play, np.where(mins > 0, rules.short_play, 0.0))
    out = out + df["goals_scored"] * pt("goals_scored") + df["assists"] * rules.pts("assists", "MID")
    out = out + np.where(mins >= 60, df["clean_sheets"] * pt("clean_sheets"), 0.0)
    out = out + np.floor(df["goals_conceded"] / 2) * pt("goals_conceded")
    out = out + np.where(pos == "GKP", np.floor(df["saves"] / 3) * rules.pts("saves", "GKP"), 0.0)
    out = out + df["penalties_saved"] * rules.pts("penalties_saved", "GKP")
    out = out + df["penalties_missed"] * rules.pts("penalties_missed", "MID")
    out = out + df["yellow_cards"] * rules.pts("yellow_cards", "MID") + df["red_cards"] * rules.pts("red_cards", "MID")
    out = out + df["own_goals"] * rules.pts("own_goals", "MID")
    if bonus:
        out = out + df["bonus"]
    if dc:
        out = out + np.where(dc_hit(pos, df["defensive_contribution"]) & (mins > 0), pt("defensive_contribution"), 0.0)
    return pd.Series(out, index=df.index, name="points")
