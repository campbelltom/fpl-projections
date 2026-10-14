"""Exact MILP for the squad objective, used to check the in-browser heuristic.  python tests/exact_squad.py predictions.json hz budget"""
import json, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BENCH_W = 0.1


def solve(path, hz, budget=100.0, time_limit=240):
    d = json.load(open(path)); need = d["squad"]
    P = [p for p in d["players"] if p["price"] is not None and p["status"] in ("a", "d")]
    n = len(P); E = np.array([p["ep"][:hz] for p in P]); price = np.round(np.array([p["price"] for p in P]) * 10)
    nv = n + 2 * n * hz                       # x_i, y_ig, c_ig
    X = lambda i: i; Y = lambda i, g: n + g * n + i; C = lambda i, g: n + n * hz + g * n + i
    c = np.zeros(nv)
    for i in range(n):
        c[X(i)] = -BENCH_W * E[i].sum()
        for g in range(hz):
            c[Y(i, g)] = -(E[i, g] - BENCH_W * E[i, g]); c[C(i, g)] = -E[i, g]
    rows, lo, hi = [], [], []
    def add(coefs, l, h): rows.append(coefs); lo.append(l); hi.append(h)
    for pos, k in need.items(): add({X(i): 1 for i, p in enumerate(P) if p["pos"] == pos}, k, k)
    add({X(i): price[i] for i in range(n)}, 0, round(budget * 10))
    for t in {p["team"] for p in P}: add({X(i): 1 for i, p in enumerate(P) if p["team"] == t}, 0, 3)
    for g in range(hz):
        add({Y(i, g): 1 for i in range(n)}, 11, 11)
        for pos, (a, b) in {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}.items():
            add({Y(i, g): 1 for i, p in enumerate(P) if p["pos"] == pos}, a, b)
        add({C(i, g): 1 for i in range(n)}, 1, 1)
        for i in range(n):
            add({Y(i, g): 1, X(i): -1}, -1, 0); add({C(i, g): 1, Y(i, g): -1}, -1, 0)
    A = lil_matrix((len(rows), nv))
    for r, co in enumerate(rows):
        for j, v in co.items(): A[r, j] = v
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    return -res.fun, res.status, [P[i]["code"] for i in range(n) if res.x[X(i)] > .5]


if __name__ == "__main__":
    print(solve(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 100.0))
