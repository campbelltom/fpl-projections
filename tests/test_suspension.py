import pandas as pd
from fplmodel.suspension import apply_ban_risk, ban_risk, reach_probs


def frame(gws=(10, 11, 12), yc90=0.3):
    return pd.DataFrame({"player_code": [1] * len(gws), "gw": list(gws), "yc90": yc90, "m_exp": 90.0,
                         "ep": 5.0, "p_play": 1.0, "pts_cards": -0.3})


def test_reach_probs_sum_and_order():
    p = reach_probs({10: 0.3, 11: 0.3}, 1)
    assert abs(sum(p.values()) - (1 - 0.7408 ** 2)) < 1e-3
    assert reach_probs({10: 0.3}, 2)[10] < p[10]


def test_ban_lands_in_next_gameweek_and_scales():
    d = frame()
    r = ban_risk(d, pd.Series({1: 4}))
    assert (1, 10) not in r and r[(1, 11)] > 0.2 and r[(1, 12)] > 0 and (1, 13) not in r
    out = apply_ban_risk(d, r)
    assert out.loc[out.gw == 10, "ep"].iloc[0] == 5.0 and out.loc[out.gw == 11, "ep"].iloc[0] < 5.0


def test_far_from_threshold_ignored_and_cutoff_respected():
    assert ban_risk(frame(), pd.Series({1: 2})) == {}
    assert ban_risk(frame((19, 20, 21)), pd.Series({1: 4})).keys() == {(1, 20)}
