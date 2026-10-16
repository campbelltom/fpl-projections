import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from deadline_gate import decide  # noqa: E402

D = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)


def run(t, concl="success"):
    return {"createdAt": t.isoformat().replace("+00:00", "Z"), "status": "completed", "conclusion": concl}


def test_before_window_does_nothing():
    assert not decide([D], D - timedelta(minutes=90), [])[0]


def test_starts_in_window_and_only_once():
    now = D - timedelta(minutes=55)
    assert decide([D], now, [])[0]
    assert not decide([D], now, [run(D - timedelta(minutes=58))])[0]          # already ran for this deadline
    assert decide([D], now, [run(D - timedelta(minutes=58), "failure")])[0]    # a failed run is retried
    assert decide([D], now, [run(D - timedelta(hours=5))])[0]                 # an older run does not count


def test_window_closes_and_next_deadline_used():
    assert not decide([D], D - timedelta(minutes=20), [])[0]
    assert not decide([D], D + timedelta(minutes=5), [])[0]
    assert decide([D, D + timedelta(days=3)], D + timedelta(days=3, minutes=-60), [run(D - timedelta(minutes=58))])[0]
