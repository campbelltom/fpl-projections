import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import fetch_prices as fp


def test_build_rows():
    boot = {"elements": [
        {"code": 111, "now_cost": 118, "price_change_percent": "-50.4", "price_change_calibrating": False, "price_change_locked_until": None,
         "price_change_projections": [{"offset": 0, "projected_percent": "-52.3", "likelihood": -3}, {"offset": 1, "projected_percent": "-60.3", "likelihood": -3}]},
        {"code": 222, "now_cost": 45, "price_change_percent": "101.0", "price_change_calibrating": True, "price_change_projections": []},
    ]}
    r = fp.build(boot)["p"]
    assert r["111"][:4] == [118, -50.4, -52.3, -3]
    assert r["111"][5] == -3 and len(r["111"]) == 8
    assert r["222"][1] == 101.0 and r["222"][2] == 101.0 and r["222"][-1] == 1   # missing projection falls back to current progress


def test_player_not_in_data_repo_falls_back_to_live_feed():
    import pandas as pd, run_production as rp
    players = pd.DataFrame({"web_name": ["Haaland"], "position": ["FWD"]}, index=pd.Index([1], name="player_code"))
    bs = {"elements": [{"code": 1, "web_name": "Haaland", "element_type": 4}, {"code": 499857, "web_name": "Newman", "element_type": 2}]}
    extra = rp.live_extras(bs, set(players.index))
    assert list(extra) == [499857]
    assert rp.player_row(players, extra, 1)["web_name"] == "Haaland"
    r = rp.player_row(players, extra, 499857)
    assert r["web_name"] == "Newman" and r["position"] == "DEF"
    assert rp.player_row(players, extra, 5) is None


def test_pending_matches_detects_unloaded_player_stats():
    import pandas as pd
    from fplmodel.players import pending_matches
    rows = [("m1", i, 90.0) for i in range(20)] + [("m2", i, 0.0) for i in range(20)] + [("m3", i, 45.0) for i in range(3)]
    pm = pd.DataFrame(rows, columns=["match_id", "player_code", "minutes_played"])
    assert pending_matches(pm, ["m1", "m2", "m3", "m4"]) == {"m2", "m3", "m4"}


def test_penalty_allocation_counts_minutes_once():
    import numpy as np
    from fplmodel.predict import pen_allocation
    m, t = np.zeros(4), np.zeros(4)
    # a sole taker who is on the pitch 87% of the time takes about 87% of the penalties, not ~100%
    p = pen_allocation(np.array([2.0, 0.02, 0.02, 0.02]), np.array([0.87, 0.6, 0.5, 0.3]), m, t)
    assert 0.84 < p[0] < 0.88
    # the penalties he misses by being off the pitch go to the others, so the team total stays near 1
    assert 0.97 < p.sum() < 1.03
    assert pen_allocation(np.array([2.0, 0.02]), np.array([1.0, 0.5]), np.zeros(2), np.zeros(2))[0] > 0.97
