import numpy as np
import pandas as pd

from fplmodel.bonus import award
from fplmodel.penalties import taker_weights


def test_bonus_tie_rules():
    cases = {
        (30, 25, 20, 10): (3, 2, 1, 0),
        (30, 30, 20, 10): (3, 3, 1, 0),      # tie for first: both 3, next gets 1
        (30, 25, 25, 10): (3, 2, 2, 0),      # tie for second: both 2, nobody gets 1
        (30, 30, 30, 10): (3, 3, 3, 0),
        (30, 25, 20, 20): (3, 2, 1, 1),      # tie for third: both 1
    }
    for bps, want in cases.items():
        assert tuple(award(np.array(bps, float))) == want, bps


def _opp(rows):
    cols = ["player_code", "team", "date", "season", "season_idx", "gw", "team_match_idx", "minutes", "pen_taken", "match_id"]
    return pd.DataFrame(rows, columns=cols)


def test_transfer_resets_penalty_evidence():
    d0 = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(10):       # player 1 took 3 penalties for club 10; player 2 took none for club 20
        rows.append((1, 10, d0 + pd.Timedelta(days=7 * i), "2025-2026", 0, i + 1, i, 90, 1 if i in (2, 5, 8) else 0, f"a{i}"))
        rows.append((2, 20, d0 + pd.Timedelta(days=7 * i), "2025-2026", 0, i + 1, i, 90, 0, f"b{i}"))
    opp = _opp(rows)
    asof = d0 + pd.Timedelta(days=200)
    idx = {10: 10, 20: 10}
    # player 1 now plays for club 20: his history does not count there, and he is flagged as a mover
    tw = taker_weights(opp, asof, 0, idx, pd.Series({1: 20, 2: 20}), pd.Series({1: np.nan, 2: 1.0}), pd.Series({1: 0.2, 2: 0.2}))
    assert tw.loc[1, "pen_ev"] == 0 and tw.loc[1, "pen_n_prev"] == 3 and bool(tw.loc[1, "mover"])
    # while at club 10 the same evidence makes him the taker
    tw2 = taker_weights(opp, asof, 0, idx, pd.Series({1: 10, 2: 20}), pd.Series({1: np.nan, 2: np.nan}), pd.Series({1: 0.2, 2: 0.2}))
    assert tw2.loc[1, "pen_share"] > 0.7 and tw2.loc[1, "pen_n_club"] == 3
