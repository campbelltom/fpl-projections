"""Central configuration. Every tunable number lives here (or in config/tuned.json,
which the backtest writes), so the model has no hidden constants."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

SEASONS = ["2024-2025", "2025-2026", "2026-2027"]
CURRENT_SEASON = "2026-2027"

# Positions as they appear in the data
POSITIONS = ["GKP", "DEF", "MID", "FWD"]

# Defensive contribution: threshold of CBIT (DEF) / CBIRT (MID, FWD) actions in a match
# that earns the DC points. Verified empirically against event_points (tests/test_scoring.py).
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": None}

HORIZON_GWS = 16

# Squad rules used for the 'value over minimum spend' number
SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0


@dataclass
class HalfLives:
    """Recency half-lives, in *matches played by the player's team* (not days), so a
    player on a team with a long break is not decayed for time the team didn't play.
    Larger = longer memory. Overwritten by config/tuned.json after the backtest."""
    xg: float = 18.0
    goals: float = 30.0
    xa: float = 18.0
    assists: float = 30.0
    shots: float = 14.0
    cbit: float = 12.0
    saves: float = 16.0
    minutes: float = 6.0
    cards: float = 40.0
    team: float = 22.0   # team attack/defence ratings


@dataclass
class Shrinkage:
    """Pseudo-observation counts used to shrink player rates toward position priors,
    and actual finishing/creation toward expected. In units of 90-minute matches (or xG)."""
    rate_matches: float = 6.0          # prior weight on position-average per-90 rates
    finishing_xg: float = 5.0          # pseudo-xG pulling goals/xG ratio to 1
    creation_xa: float = 5.0           # pseudo-xA pulling assists/xA ratio to 1
    prior_season_weight: float = 0.7   # weight multiplier on previous-season matches
    team_ridge: float = 1.2            # L2 strength on team log-factors (toward prior)


@dataclass
class Params:
    hl: HalfLives = field(default_factory=HalfLives)
    shrink: Shrinkage = field(default_factory=Shrinkage)
    # Extra tunables
    home_adv_floor: float = 1.0
    nb_dispersion_dc: float = 0.08     # negative-binomial dispersion for DC action counts
    version: str = "0.1"

    @classmethod
    def load(cls) -> "Params":
        p = cls()
        f = CONFIG_DIR / "tuned.json"
        if f.exists():
            d = json.loads(f.read_text())
            for k, v in d.get("half_lives", {}).items():
                if hasattr(p.hl, k):
                    setattr(p.hl, k, float(v))
            for k, v in d.get("shrinkage", {}).items():
                if hasattr(p.shrink, k):
                    setattr(p.shrink, k, float(v))
            for k in ("nb_dispersion_dc",):
                if k in d:
                    setattr(p, k, float(d[k]))
        return p
