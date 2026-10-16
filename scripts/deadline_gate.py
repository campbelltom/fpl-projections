"""Run the update an hour before each FPL deadline.

A small workflow (deadline.yml) calls this every 10 minutes. It finds the next deadline, and when we are inside
the window [deadline - LEAD, deadline - LEAD + WINDOW) and no update has started since the window opened, it
starts update.yml. Everything is stdlib so the check costs a few seconds.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

LEAD = timedelta(minutes=60)        # update this long before the deadline
WINDOW = timedelta(minutes=30)      # tolerate late cron ticks (GitHub can delay schedules)
try:  # keep the season in one place when the package is importable
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from fplmodel.config import CURRENT_SEASON as SEASON
except Exception:  # noqa: BLE001
    SEASON = "2026-2027"
API = "https://fantasy.premierleague.com/api/bootstrap-static/"
CSV = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/{SEASON}/gameweek_summaries.csv"
UA = {"User-Agent": "Mozilla/5.0 (campbell-fpl deadline gate)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def deadlines() -> list[datetime]:
    try:
        ev = json.loads(_get(API))["events"]
        out = [parse(e["deadline_time"]) for e in ev if e.get("deadline_time")]
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        print(f"FPL API unavailable ({e}); using the data repo", file=sys.stderr)
    rows = csv.DictReader(io.StringIO(_get(CSV).decode()))
    return [parse(r["deadline_time"]) for r in rows if r.get("deadline_time")]


def decide(dls: list[datetime], now: datetime, runs: list[dict]) -> tuple[bool, str]:
    """runs: recent update.yml runs as {createdAt, status, conclusion}. Returns (start?, reason)."""
    nxt = min((d for d in dls if d > now), default=None)
    if nxt is None:
        return False, "no upcoming deadline"
    t0 = nxt - LEAD
    if now < t0:
        return False, f"next deadline {nxt:%a %d %b %H:%M} UTC; window opens {t0:%H:%M}"
    if now >= t0 + WINDOW:
        return False, f"window for the {nxt:%a %d %b %H:%M} deadline already passed"
    for r in runs:
        started = parse(r["createdAt"])
        if started >= t0 and r.get("conclusion") not in ("failure", "cancelled", "timed_out"):
            return False, f"already updated at {started:%H:%M} for this deadline"
    return True, f"deadline {nxt:%a %d %b %H:%M} UTC is {int((nxt - now).total_seconds() // 60)} minutes away"


def recent_runs() -> list[dict]:
    out = subprocess.run(["gh", "run", "list", "--workflow", "update.yml", "--limit", "20", "--json", "createdAt,status,conclusion"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> int:
    now = datetime.now(timezone.utc)
    go, why = decide(deadlines(), now, recent_runs())
    print(("START: " if go else "skip: ") + why)
    if go:
        subprocess.run(["gh", "workflow", "run", "update.yml"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
