"""Fetch a manager's current FPL squad (public endpoints, no login) and turn it into what the squad builder needs:
squad with SELLING prices, bank, and free transfers.  Network access is isolated in `fetch_json` so everything else is testable."""
from __future__ import annotations

import json
import time
import urllib.request

API = "https://fantasy.premierleague.com/api"


def fetch_json(path: str, tries: int = 3) -> dict | list:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "campbell-fpl/2.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{path}: {last}")


def sell_price(now: float, purchase: float) -> float:
    """FPL sells at the purchase price plus half the profit, rounded down to 0.1m; losses are taken in full."""
    profit = round((now - purchase) * 10)
    return round((purchase * 10 + (profit // 2 if profit > 0 else profit)) / 10, 1)


def free_transfers(history: dict, gw0: int, cap: int = 5) -> int:
    """Estimated free transfers available for `gw0`, replaying the season. Chips (wildcard / free hit) do not consume
    transfers. This is an estimate: the site lets the user correct it."""
    used = {h["event"]: h.get("event_transfers", 0) for h in history.get("current", [])}
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    ft = 1                                           # entering GW2
    for e in range(2, gw0):
        n = 0 if chips.get(e) in ("wildcard", "freehit", "3xc_wc") else used.get(e, 0)
        ft = min(max(ft - n, 0) + 1, cap) if n <= ft else 1
    return ft


def build_team(entry_id: int, label: str, gw0: int, id2code: dict[int, int], now: dict[int, float], start: dict[int, float],
               get=fetch_json) -> dict:
    """gw0 = the upcoming gameweek. Uses the last finished gameweek's picks plus any transfers already made for gw0."""
    entry = get(f"entry/{entry_id}/")
    hist = get(f"entry/{entry_id}/history/")
    transfers = get(f"entry/{entry_id}/transfers/")
    last = gw0 - 1
    if last < 1:
        raise RuntimeError("season has not started: no picks yet")
    # a free hit squad is not kept: the team (and bank) go back to the gameweek before, and its purchases do not count
    fh = {c["event"] for c in hist.get("chips", []) if c.get("name") == "freehit" and c["event"] != gw0}
    while last in fh and last > 1:
        last -= 1
    picks = get(f"entry/{entry_id}/event/{last}/picks/")
    squad = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0
    tr = sorted((t for t in transfers if t["event"] not in fh), key=lambda t: t["time"])
    pending = [t for t in tr if t["event"] == gw0]
    for t in pending:                                # transfers already made for the upcoming gameweek
        if t["element_out"] in squad:
            squad.remove(t["element_out"])
        squad.append(t["element_in"])
        bank += (t["element_out_cost"] - t["element_in_cost"]) / 10.0
    purchase = {}
    for t in tr:                                     # most recent purchase of each player
        purchase[t["element_in"]] = t["element_in_cost"] / 10.0
    out = []
    for el in squad:
        code = id2code.get(el)
        if code is None:
            raise RuntimeError(f"unknown player id {el}")
        n = now[code]
        p = purchase.get(el, start.get(code, n))
        out.append({"code": int(code), "sell": sell_price(n, p), "now": n})
    ft = max(free_transfers(hist, gw0) - len(pending), 0)
    return {"id": entry_id, "label": label, "name": entry.get("name"), "manager": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
            "gw": last, "bank": round(bank, 1), "ft": ft, "pending": len(pending), "squad": out,
            "points": entry.get("summary_overall_points"), "rank": entry.get("summary_overall_rank")}
