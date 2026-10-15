"""Fit how Bonus Points System (BPS) scores arise from match events, per position -> config/bps_model.json.

BPS = const + b60*(60+ minutes) + sum(weight * event) + other,  other ~ N(0, sd).
The weights come out close to FPL's published values (goal 12/18/24, assist 9, clean sheet 12, goal conceded -4 ...),
absorbing the stats FPL scores that we cannot forecast (passes, key passes, tackles) into `dc`, `xa`, `xg`.

   python scripts/fit_bps.py <source_dir>
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.config import CONFIG_DIR  # noqa: E402
from fplmodel.data import load_all  # noqa: E402

FEATS = {"i60": None, "goals_scored": "goals", "assists": "assists", "clean_sheets": "cs", "goals_conceded": "gc", "saves": "saves",
         "yellow_cards": "yc", "red_cards": "rc", "own_goals": "og", "penalties_missed": "pm", "penalties_saved": "ps",
         "defensive_contribution": "dc", "expected_assists": "xa", "expected_goals": "xg"}


def main(src: str) -> None:
    S = load_all(src, ["2025-2026", "2026-2027"])
    rows = []
    for n, s in S.items():
        g = s.pgw.copy()
        g["position"] = g["player_code"].map(s.players.set_index("player_code")["position"])
        nm = s.pms[s.pms.minutes_played > 0].groupby(["player_code", "gw"]).size().rename("nm").reset_index()
        rows.append(g.merge(nm, on=["player_code", "gw"], how="left"))
    g = pd.concat(rows)
    g = g[(g.nm == 1) & (g.minutes > 0)].copy()
    g["i60"] = (g.minutes >= 60).astype(float)
    out = {}
    for p, h in g.groupby("position"):
        cols = list(FEATS)
        A = np.column_stack([np.ones(len(h))] + [h[c].fillna(0).to_numpy(float) for c in cols])
        y = h["bps"].to_numpy(float)
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ b
        out[p] = {"const": float(b[0]), **{FEATS[c] or c: float(v) for c, v in zip(cols, b[1:])}, "sd": float(r.std()),
                  "n": int(len(h)), "r2": float(1 - r.var() / y.var())}
        print(p, len(h), "R2", round(out[p]["r2"], 3), "sd", round(out[p]["sd"], 2))
    (CONFIG_DIR / "bps_model.json").write_text(json.dumps(out, indent=1))
    print("wrote", CONFIG_DIR / "bps_model.json")


if __name__ == "__main__":
    main(sys.argv[1])
