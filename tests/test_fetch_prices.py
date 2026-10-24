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
