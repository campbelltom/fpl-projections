"""Full walk-forward backtest with diagnostics.   python scripts/run_backtest.py <source_dir> [--quick]"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fplmodel.backtest import metrics, walk_forward  # noqa: E402
from fplmodel.config import Params  # noqa: E402
from fplmodel.engine import Context  # noqa: E402
from fplmodel.scoring import load_rules  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 40)


def make_plan(quick: bool) -> list[tuple[str, int]]:
    plan = [("2025-2026", g) for g in range(6, 39, 3 if quick else 1)]
    plan += [("2026-2027", g) for g in range(2, 7)]
    return plan


def report(res: pd.DataFrame, label: str = "") -> None:
    d = res.dropna(subset=["event_points"]).copy()
    d = d[d["gw"].notna()]
    fpl = d.dropna(subset=["ep_next"])
    print(f"== {label} rows={len(d)}")
    print("  model      ", {k: round(v, 3) for k, v in metrics(d).items()})
    print("  fpl ep_next", {k: round(v, 3) for k, v in metrics(fpl.assign(x=fpl.ep_next), "x").items()},
          "(same rows as model:", {k: round(v, 3) for k, v in metrics(fpl).items()}, ")")
    print("  by position (MAE / bias / mean pred vs actual):")
    for p, g in d.groupby("position"):
        print(f"    {p}: n={len(g)} MAE {np.abs(g.ep-g.event_points).mean():.3f} bias {(g.ep-g.event_points).mean():+.3f}"
              f"  pred {g.ep.mean():.3f} actual {g.event_points.mean():.3f}")
    d["bin"] = pd.cut(d["ep"], [-1, 0.5, 1, 2, 3, 4, 5, 6, 8, 20])
    c = d.groupby("bin", observed=True).agg(n=("ep", "size"), pred=("ep", "mean"), actual=("event_points", "mean"))
    print("  calibration by predicted EP bin:")
    print(c.round(3).to_string())
    played = (d["minutes"] > 0)
    print(f"  P(play): predicted mean {d.p_play.mean():.3f} vs actual {played.mean():.3f}")
    d["pbin"] = pd.cut(d["p_play"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
    print(d.groupby("pbin", observed=True).agg(n=("p_play", "size"), pred=("p_play", "mean"), actual=("minutes", lambda x: (x > 0).mean())).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ctx = Context.build(a.src)
    res = walk_forward(ctx, Params.load(), load_rules(live=False), make_plan(a.quick))
    report(res, "all")
    report(res[res.season == "2025-2026"], "2025-26")
    report(res[res.season == "2026-2027"], "2026-27 (6 GWs)")
    res.to_pickle(Path(__file__).resolve().parent.parent / "config" / "_last_backtest.pkl")
