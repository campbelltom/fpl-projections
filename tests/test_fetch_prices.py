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
