import numpy as np
import pandas as pd

from fplmodel.fxfeat import calendar_map, fixture_features, rival_features


def test_rest_days_use_every_competition():
    cal = pd.DataFrame({"team": [1, 1, 1, 1],
                        "date": pd.to_datetime(["2026-09-01", "2026-09-05", "2026-09-08", "2026-09-20"], utc=True)})
    cm = calendar_map(cal)
    f = fixture_features(cm, pd.Series([1]), pd.Series(pd.to_datetime(["2026-09-05"], utc=True)), None, pd.Series([1]))
    assert f["rest_b"].iloc[0] == 4 and f["rest_a"].iloc[0] == 3 and f["n14"].iloc[0] == 1


def test_rivals_exclude_self():
    es = pd.Series([0.9, 0.8, 0.1])
    key = pd.DataFrame({"m": [1, 1, 1], "t": [1, 1, 1], "p": ["MID"] * 3})
    r = rival_features(es, key)
    assert np.isclose(r["rv_max"].iloc[0], 0.8) and np.isclose(r["rv_max"].iloc[1], 0.9)
    assert np.isclose(r["rv_sum"].iloc[2], 1.7)
