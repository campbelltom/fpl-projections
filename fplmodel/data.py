"""Load FPL-Core-Insights into clean, stable-keyed tables.

Hard-won rules (each one is a bug the old build could have had):
  * player_id and team id are NOT stable across seasons -> key on player_code / team code.
  * 2025/26 early gameweeks only list players who appeared; later ones list unused subs
    with 0 minutes. Never infer minutes from presence alone.
  * chance_of_playing_next_round is blank for fit players in 2026/27 but 0 for fit players in
    some 2025/26 snapshots -> availability keys on `status`, never on that column for 'a'.
  * 'current gameweek' comes from the fixture list, not gameweek_summaries.csv (stale flags).
  * start_min/finish_min are unreliable -> minutes_played is the source of truth.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEASONS

import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

log = logging.getLogger("fplmodel.data")


POS_MAP = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD",
           "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def _gw_of(path: str) -> int:
    m = re.search(r"GW(\d+)", path)
    return int(m.group(1)) if m else -1


def _read_gw_folder(pattern: str) -> pd.DataFrame:
    """Concatenate per-gameweek CSVs, adding a `gw` column from the folder name."""
    frames = []
    for p in sorted(glob.glob(pattern), key=_gw_of):
        try:
            df = pd.read_csv(p, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        frames.append(df.assign(gw=_gw_of(p)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class Season:
    name: str
    players: pd.DataFrame          # player_code, player_id, web_name, position, team_code
    teams: pd.DataFrame            # code, id, name, short_name
    matches: pd.DataFrame          # one row per PL match (finished or not), team CODES
    pms: pd.DataFrame              # player-match stats, player_code + team_code + gw
    pgw: pd.DataFrame              # per-gameweek player stats (event_points, bonus, ...)
    snapshot: pd.DataFrame         # latest playerstats snapshot (status/news/ep_next)
    team_hist: pd.DataFrame = field(default_factory=pd.DataFrame)   # player_code, gw, team_code (registration)
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)    # team, date: every match a PL club plays, all competitions
    notes: list[str] = field(default_factory=list)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def load_season(src: str | Path, season: str) -> Season:
    src = Path(src)
    d = src / "data" / season
    notes: list[str] = []
    if not d.exists():
        raise FileNotFoundError(d)

    nested = (d / "players" / "players.csv").exists()        # 2024-25 layout
    if nested:
        players = pd.read_csv(d / "players" / "players.csv")
        teams = pd.read_csv(d / "teams" / "teams.csv")
        matches = _read_gw_folder(str(d / "matches" / "GW*" / "matches.csv"))
        pms = _read_gw_folder(str(d / "playermatchstats" / "GW*" / "playermatchstats.csv"))
        pgw = pd.DataFrame()
        snapshot = pd.DataFrame()
        team_hist = pd.DataFrame()
    else:
        base = d / "By Tournament" / "Premier League"
        players = pd.read_csv(d / "players.csv")
        teams = pd.read_csv(d / "teams.csv")
        matches = _read_gw_folder(str(base / "GW*" / "matches.csv"))
        pms = _read_gw_folder(str(base / "GW*" / "playermatchstats.csv"))
        pgw = _read_gw_folder(str(base / "GW*" / "player_gameweek_stats.csv"))
        snap = _read_gw_folder(str(d / "By Gameweek" / "GW*" / "playerstats.csv"))
        if not snap.empty:
            snap = snap[snap.gw == snap.gw.max()].copy()
        snapshot = snap
        th = d / "team_history.csv"
        team_hist = pd.read_csv(th) if th.exists() else pd.DataFrame()

    # ---- id -> code maps for THIS season
    tid2code = dict(zip(teams["id"].astype(int), teams["code"].astype(int)))
    pid2code = dict(zip(players["player_id"].astype(int), players["player_code"].astype(int)))
    players = players.copy()
    players["team_code"] = players["team_code"].astype("Int64")
    players["position"] = players["position"].map(POS_MAP).where(lambda s: s.notna(), None)

    # ---- matches: dedupe on match_id (prefer finished), resolve teams to CODES.
    # The README says home_team/away_team are team ids, but the data holds team codes.
    # Detect which key space each season uses and insist on >=99% resolution.
    m = matches.copy()
    m["finished"] = _bool(m["finished"])
    m = m.sort_values(["match_id", "finished"], ascending=[True, False]).drop_duplicates("match_id", keep="first")
    ht, at = m["home_team"].astype("Int64"), m["away_team"].astype("Int64")
    codes = set(tid2code.values())
    frac_code = float(pd.concat([ht, at]).isin(list(codes)).mean())
    frac_id = float(pd.concat([ht, at]).isin(list(tid2code)).mean())
    if frac_code >= frac_id:
        m["home_code"], m["away_code"], space, frac = ht, at, "code", frac_code
    else:
        m["home_code"] = ht.map(lambda v: tid2code.get(int(v)) if pd.notna(v) else None)
        m["away_code"] = at.map(lambda v: tid2code.get(int(v)) if pd.notna(v) else None)
        space, frac = "id", frac_id
    if frac < 0.99:
        raise ValueError(f"{season}: only {frac:.1%} of match team keys resolve ({space} space)")
    notes.append(f"{season}: match team keys are team {space}s ({frac:.1%} resolved)")
    bad = m[["home_code", "away_code"]].isna().any(axis=1).sum()
    if bad:
        notes.append(f"{season}: {bad} matches with unresolvable teams dropped")
        m = m.dropna(subset=["home_code", "away_code"])
    m[["home_code", "away_code"]] = m[["home_code", "away_code"]].astype(int)
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"], utc=True, errors="coerce")
    m["season"] = season
    m = m.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)

    # ---- player-match rows: player_code, team_code
    p = pms.copy()
    p["player_code"] = p["player_id"].astype(int).map(pid2code)
    unk = p["player_code"].isna().sum()
    if unk:
        notes.append(f"{season}: {unk} player-match rows with unknown player_id dropped")
        p = p.dropna(subset=["player_code"])
    p["player_code"] = p["player_code"].astype(int)

    mm = m[["match_id", "home_code", "away_code", "kickoff_time", "finished"]]
    p = p.merge(mm, on="match_id", how="inner")
    p = p[p["finished"]].copy()

    # team of the player in that match: team_history (by gw) -> players.csv fallback,
    # and ALWAYS validated against the match's two teams.
    num = lambda s: pd.to_numeric(s, errors="coerce").astype("float64")  # noqa: E731
    p["team_code"] = np.nan
    if not team_hist.empty:
        th = team_hist.copy()
        th["player_code"] = th["player_id"].astype(int).map(pid2code)
        th = th.dropna(subset=["player_code"]).drop_duplicates(["player_code", "gw"], keep="last")
        th["player_code"] = th["player_code"].astype(int)
        p = p.merge(th[["player_code", "gw", "team_code"]].rename(columns={"team_code": "team_hist"}),
                    on=["player_code", "gw"], how="left")
        p["team_code"] = num(p["team_hist"])
        p = p.drop(columns=["team_hist"])
    fallback = num(p["player_code"].map(dict(zip(players["player_code"].astype(int),
                                                  num(players["team_code"])))))
    in_match = lambda col: (col == p["home_code"]) | (col == p["away_code"])  # noqa: E731
    use_fb = ~in_match(p["team_code"]) & in_match(fallback)
    p.loc[use_fb, "team_code"] = fallback[use_fb]
    ok = in_match(p["team_code"])
    if (~ok).sum():
        notes.append(f"{season}: {(~ok).sum()} player-match rows could not be placed on a team and were dropped "
                     f"({(~ok).mean():.2%})")
    p = p[ok].copy()
    p["team_code"] = p["team_code"].astype(int)
    p["opp_code"] = np.where(p["team_code"] == p["home_code"], p["away_code"], p["home_code"])
    p["is_home"] = (p["team_code"] == p["home_code"]).astype(int)
    p["season"] = season
    # one row per player per match (guard against duplicates)
    dup = p.duplicated(["player_code", "match_id"]).sum()
    if dup:
        notes.append(f"{season}: {dup} duplicate player-match rows removed")
        p = p.drop_duplicates(["player_code", "match_id"], keep="last")

    # ---- harmonise attribute definitions across seasons
    p["position"] = p["player_code"].map(dict(zip(players["player_code"].astype(int), players["position"])))
    for c in ["tackles", "tackles_won", "interceptions", "blocks", "clearances", "recoveries"]:
        if c not in p:
            p[c] = 0.0
    played = p["minutes_played"].fillna(0) > 0
    tk_att = p.loc[played, "tackles"].fillna(0).sum()
    tk_won = p.loc[played, "tackles_won"].fillna(0).sum()
    # 2026/27 stopped populating `tackles`, and `tackles_won` took over its meaning (it is the
    # figure that matches FPL's own DC count). Use whichever is populated, never mix them.
    if tk_att >= 0.3 * max(tk_won, 1):
        p["tackles_h"] = p["tackles"].fillna(0)
        notes.append(f"{season}: tackle count = `tackles` (attempted)")
    else:
        p["tackles_h"] = p["tackles_won"].fillna(0)
        notes.append(f"{season}: `tackles` unpopulated; tackle count = `tackles_won`")
    p["cbit"] = p["tackles_h"] + p[["interceptions", "blocks", "clearances"]].fillna(0).sum(axis=1)
    p["cbirt"] = p["cbit"] + p["recoveries"].fillna(0)
    # the count FPL thresholds against: CBIT for defenders, CBIRT for midfielders/forwards
    p["dc_actions"] = np.where(p["position"] == "DEF", p["cbit"],
                               np.where(p["position"].isin(["MID", "FWD"]), p["cbirt"], 0.0))
    # column health: attributes that are present but unpopulated for the whole season
    for c in ["xg", "xa", "total_shots", "shots_on_target", "saves", "xgot_faced", "big_chances_missed", "chances_created"]:
        if c in p and played.sum() and p.loc[played, c].fillna(0).abs().sum() == 0:
            notes.append(f"{season}: column `{c}` is all zero/blank - treated as missing")

    # ---- per-gameweek player table
    if not pgw.empty:
        g = pgw.copy()
        g["player_code"] = g["id"].astype(int).map(pid2code)
        g = g.dropna(subset=["player_code"])
        g["player_code"] = g["player_code"].astype(int)
        g["season"] = season
    else:
        g = pgw
    if not snapshot.empty:
        s = snapshot.copy()
        s["player_code"] = s["id"].astype(int).map(pid2code)
        s = s.dropna(subset=["player_code"])
        s["player_code"] = s["player_code"].astype(int)
        snapshot = s

    pl = players.copy()
    pl["player_code"] = pl["player_code"].astype(int)
    # registration of each player to a team by gameweek
    if not team_hist.empty:
        th = team_hist.copy()
        th["player_code"] = th["player_id"].astype(int).map(pid2code)
        th = th.dropna(subset=["player_code"])
        th["player_code"] = th["player_code"].astype(int)
        th = th.drop_duplicates(["player_code", "gw"], keep="last")[["player_code", "gw", "team_code"]]
    else:   # 2024-25 has no history file: use the roster team for every gameweek
        gws = sorted(m["gameweek"].dropna().astype(int).unique())
        base = pl[["player_code", "team_code"]].dropna()
        th = pd.concat([base.assign(gw=g) for g in gws], ignore_index=True)
    th["team_code"] = th["team_code"].astype(int)
    # ---- club calendar across every competition (rest days and rotation risk)
    rows = [m[["home_code", "kickoff_time"]].rename(columns={"home_code": "team", "kickoff_time": "date"}),
            m[["away_code", "kickoff_time"]].rename(columns={"away_code": "team", "kickoff_time": "date"})]
    if not nested:
        tdir = d / "By Tournament"
        for sub in sorted(tdir.iterdir()) if tdir.exists() else []:
            if sub.name in ("Premier League", "Friendlies", "Community Shield", "Uefa Super Cup") or not sub.is_dir():
                continue
            om = _read_gw_folder(str(sub / "GW*" / "matches.csv"))
            if om.empty or "kickoff_time" not in om:
                continue
            for side in ("home_team", "away_team"):
                v = pd.to_numeric(om[side], errors="coerce")
                c = v.where(v.isin(list(codes))) if space == "code" else v.map(lambda x: tid2code.get(int(x)) if pd.notna(x) and int(x) in tid2code else np.nan)
                rows.append(pd.DataFrame({"team": c, "date": pd.to_datetime(om["kickoff_time"], utc=True, errors="coerce")}))
    cal = pd.concat(rows, ignore_index=True).dropna()
    cal["team"] = cal["team"].astype(int)
    cal = cal.drop_duplicates().sort_values(["team", "date"]).reset_index(drop=True)
    return Season(season, pl, teams.assign(code=teams["code"].astype(int)), m, p, g, snapshot, th, calendar=cal, notes=notes)


def load_all(src: str | Path, seasons: list[str] | None = None) -> dict[str, Season]:
    out = {}
    for s in (seasons or SEASONS):
        try:
            out[s] = load_season(src, s)
        except FileNotFoundError:
            log.warning("season %s missing from source", s)
    return out


def current_gw(matches: pd.DataFrame) -> int | None:
    """First gameweek containing an unfinished match = the next gameweek to be played."""
    un = matches[~matches["finished"]]
    return int(un["gameweek"].min()) if len(un) else None
