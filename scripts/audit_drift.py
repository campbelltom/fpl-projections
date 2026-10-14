"""Audit: per-90 means of every player-match attribute by position and season.
A definition change between seasons (like the tackles columns) shows up as a step change."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.data import load_all  # noqa: E402

ATTRS = ["goals", "assists", "xg", "xa", "xgot", "total_shots", "shots_on_target", "chances_created", "touches",
         "tackles", "tackles_won", "interceptions", "blocks", "clearances", "recoveries", "duels_won", "duels_lost",
         "was_fouled", "fouls_committed", "saves", "goals_conceded", "xgot_faced", "accurate_passes", "big_chances_missed"]


def main(src: str) -> None:
    S = load_all(src)
    rows = []
    for k, s in S.items():
        pos = s.players.set_index("player_code")["position"]
        p = s.pms[s.pms.minutes_played >= 30].copy()
        p["position"] = p.player_code.map(pos)
        for pp, grp in p.groupby("position"):
            r = {"season": k, "pos": pp, "n": len(grp)}
            for a in ATTRS:
                if a in grp:
                    r[a] = (grp[a].sum() / grp.minutes_played.sum()) * 90
            rows.append(r)
    df = pd.DataFrame(rows).set_index(["pos", "season"]).sort_index()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "source")
