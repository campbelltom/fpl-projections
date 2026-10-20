"""Fetch the official FPL price-change predictor figures and write a small public prices.json.

The Fantasy site's own predictor is part of the bootstrap-static feed: for each player, price_change_percent is the progress towards the
next rise (towards +100) or fall (towards -100), and price_change_projections gives the projected progress at the next three daily
changes (00:00 UK) with a likelihood from -3 (very likely to fall) to +3 (very likely to rise). Prices are public data.

Output: {"at": iso time, "p": {player_code: [now_cost, pct, proj0, like0, proj1, like1, proj2, like2]}}
"""
import json, sys, datetime as dt, urllib.request

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build(boot, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    out = {}
    for e in boot.get("elements", []):
        proj = {p.get("offset"): p for p in (e.get("price_change_projections") or [])}
        row = [int(e["now_cost"]), round(_num(e.get("price_change_percent")), 1)]
        for off in (0, 1, 2):
            p = proj.get(off) or {}
            row += [round(_num(p.get("projected_percent"), row[1]), 1), int(_num(p.get("likelihood")))]
        if e.get("price_change_calibrating") or e.get("price_change_locked_until"):
            row.append(1)          # 9th value: 1 = the figures are not reliable yet
        out[str(e["code"])] = row
    return {"at": now.replace(microsecond=0).isoformat(), "p": out}


def main(dest):
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        boot = json.load(r)
    res = build(boot)
    if len(res["p"]) < 300:
        raise SystemExit("too few players in the feed; not writing")
    with open(dest, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print(f"wrote {len(res['p'])} players to {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prices.json")
