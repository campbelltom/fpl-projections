import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.myteam import build_team, free_transfers, sell_price


def test_sell_price_rounds_profit_down_and_takes_losses_in_full():
    assert sell_price(5.5, 5.0) == 5.2      # +0.5 profit -> +0.2 (floor of half in tenths)
    assert sell_price(5.0, 5.0) == 5.0
    assert sell_price(4.7, 5.0) == 4.7      # losses in full
    assert sell_price(5.1, 5.0) == 5.0      # +0.1 profit -> nothing


def test_free_transfers_bank_and_cap():
    h = {"current": [{"event": e, "event_transfers": 0} for e in range(1, 7)], "chips": []}
    assert free_transfers(h, 7) == 5        # saved every week from GW2: 1 -> capped at 5
    h2 = {"current": [{"event": 2, "event_transfers": 1}, {"event": 3, "event_transfers": 0}], "chips": []}
    assert free_transfers(h2, 4) == 2
    h3 = {"current": [{"event": 2, "event_transfers": 3}], "chips": []}
    assert free_transfers(h3, 3) == 1       # took a hit: back to one
    h4 = {"current": [{"event": 2, "event_transfers": 15}], "chips": [{"event": 2, "name": "wildcard"}]}
    assert free_transfers(h4, 3) == 2       # wildcard does not consume transfers


def test_build_team_applies_pending_transfer_and_selling_price():
    id2code = {i: 1000 + i for i in range(1, 17)}
    now = {1000 + i: 5.0 for i in range(1, 17)}; now[1001] = 5.5
    start = {c: 5.0 for c in now}
    data = {
        "entry/9/": {"name": "T", "player_first_name": "A", "player_last_name": "B"},
        "entry/9/history/": {"current": [{"event": 2, "event_transfers": 0}], "chips": []},
        "entry/9/transfers/": [{"event": 3, "time": "2026-10-10T10:00:00Z", "element_in": 16, "element_in_cost": 50,
                                 "element_out": 15, "element_out_cost": 50}],
        "entry/9/event/2/picks/": {"entry_history": {"bank": 10}, "picks": [{"element": i} for i in range(1, 16)]},
    }
    t = build_team(9, "x", 3, id2code, now, start, get=lambda p: data[p])
    codes = {p["code"] for p in t["squad"]}
    assert 1016 in codes and 1015 not in codes and len(t["squad"]) == 15
    assert [p for p in t["squad"] if p["code"] == 1001][0]["sell"] == 5.2
    assert t["bank"] == 1.0 and t["pending"] == 1 and t["ft"] == 1   # 2 banked... minus the pending one


def test_free_hit_week_is_ignored_for_squad_bank_and_purchase_prices():
    id2code = {i: 1000 + i for i in range(1, 18)}
    now = {1000 + i: 5.0 for i in range(1, 18)}; now[1001] = 5.5
    start = {c: 5.0 for c in now}
    data = {
        "entry/9/": {"name": "T"},
        "entry/9/history/": {"current": [{"event": 2, "event_transfers": 0}, {"event": 3, "event_transfers": 15}], "chips": [{"event": 3, "name": "freehit"}]},
        # during the free hit he bought player 1 (at 5.5) and 17; neither stays
        "entry/9/transfers/": [{"event": 3, "time": "2026-10-10T10:00:00Z", "element_in": 17, "element_in_cost": 55, "element_out": 1, "element_out_cost": 50},
                                {"event": 3, "time": "2026-10-10T10:01:00Z", "element_in": 1, "element_in_cost": 55, "element_out": 2, "element_out_cost": 50}],
        "entry/9/event/2/picks/": {"entry_history": {"bank": 10}, "picks": [{"element": i} for i in range(1, 16)]},
        "entry/9/event/3/picks/": {"entry_history": {"bank": 0}, "picks": [{"element": i} for i in range(3, 18)]},
    }
    t = build_team(9, "x", 4, id2code, now, start, get=lambda p: data[p])
    assert {p["code"] for p in t["squad"]} == {1000 + i for i in range(1, 16)}      # the pre-free-hit squad
    assert t["bank"] == 1.0
    assert [p for p in t["squad"] if p["code"] == 1001][0]["sell"] == 5.2         # bought at 5.0 originally, not at the free hit price
