import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.export import encrypt_json
from fplmodel.predict import e_floor_div
import numpy as np


def test_encrypt_roundtrip_shape():
    pkg = encrypt_json({"a": 1}, "pw")
    assert set(pkg) == {"v", "iter", "salt", "iv", "ct"} and pkg["iter"] >= 200_000
    assert encrypt_json({"a": 1}, "pw")["ct"] != pkg["ct"]   # fresh salt/iv every run


def test_floor_div_expectation():
    # E[floor(X/2)] for X ~ Poisson(lam) is below lam/2 and rises with lam
    a, b = float(e_floor_div(np.array([1.0]), 2)[0]), float(e_floor_div(np.array([2.0]), 2)[0])
    assert 0 < a < 0.5 < b < 1.0
