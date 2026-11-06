import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_live as L  # noqa: E402


def fx(i, th, ta, started=True, fin=False, stats=None, ko="2026-11-07T15:00:00Z"):
    return {"id": i, "team_h": th, "team_a": ta, "started": started, "finished": fin, "finished_provisional": fin, "minutes": 60,
            "kickoff_time": ko, "team_h_score": 1, "team_a_score": 0, "stats": stats or []}


def test_provisional_bonus_with_ties():
    stats = [{"identifier": "bps", "h": [{"element": 1, "value": 40}, {"element": 2, "value": 40}], "a": [{"element": 3, "value": 30}, {"element": 4, "value": 20}]}]
    b = L.bonus_by_fixture([fx(1, 1, 2, stats=stats)])[1]
    assert b == {1: 3, 2: 3, 3: 1}


def test_official_bonus_wins_and_unstarted_ignored():
    stats = [{"identifier": "bonus", "h": [{"element": 9, "value": 2}], "a": []}, {"identifier": "bps", "h": [{"element": 1, "value": 50}], "a": []}]
    assert L.bonus_by_fixture([fx(1, 1, 2, stats=stats)])[1] == {9: 2}
    assert L.bonus_by_fixture([fx(2, 1, 2, started=False)]) == {}


def test_window_gate():
    now = dt.datetime(2026, 11, 7, 16, 0, tzinfo=dt.timezone.utc)
    assert L.in_window([fx(1, 1, 2)], now)
    assert not L.in_window([fx(1, 1, 2, fin=True, ko="2026-11-07T08:00:00Z")], now)
    assert not L.in_window([fx(1, 1, 2, started=False, ko="2026-11-08T15:00:00Z")], now)


POS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 1, 13: 2, 14: 3, 15: 4}


def test_auto_sub_keeps_formation():
    xi = list(range(1, 12))
    # a defender (5) didn't play; the bench keeper can't replace him, the next outfield player can
    played = lambda p: p != 5
    out, swaps = L.auto_subs(xi, [12, 13, 14, 15], played, lambda p: True, POS)
    assert swaps == [[5, 13]] and 13 in out and 5 not in out
    # with only 3 defenders left a defender can't be swapped for a midfielder
    xi2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    played2 = lambda p: p not in (5, 12 + 9)
    out2, sw2 = L.auto_subs(xi2, [12, 14, 15, 13], played2, lambda p: True, POS)
    assert sw2 == [[5, 14]] or sw2 == [[5, 13]]
    # a keeper is only replaced by the bench keeper
    out3, sw3 = L.auto_subs(xi, [13, 14, 15, 12], lambda p: p != 1, lambda p: True, POS)
    assert sw3 == [[1, 12]]


def test_not_done_means_no_sub():
    assert L.auto_subs(list(range(1, 12)), [12, 13, 14, 15], lambda p: p != 5, lambda p: False, POS)[1] == []
