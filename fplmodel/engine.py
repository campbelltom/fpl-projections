"""Orchestration: build all tables once, then predict as of any date (live or historical)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .availability import AvailabilityModel, HORIZON
from .config import Params
from .data import Season, load_all
from .minutes import MinutesModel, add_state, features_from_state, latest_state
from .penalties import taker_weights, team_rates, team_summary
from .players import build_opportunities, player_profiles
from .predict import BonusModel, aggregate_gameweek, expected_points, fixture_context
from .scoring import Rules, load_rules
from .teams import build_team_matches, fit_team_model

log = logging.getLogger("fplmodel.engine")


@dataclass
class Context:
    seasons: dict[str, Season]
    tm: pd.DataFrame
    opp: pd.DataFrame
    status: pd.DataFrame
    gw_date: pd.DataFrame
    actual: pd.DataFrame
    state: pd.DataFrame = None
    norm: tuple = (9.3, 14.0)
    names: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, src, seasons: list[str] | None = None) -> "Context":
        S = load_all(src, seasons)
        names = list(S)
        tm = build_team_matches(S)
        opp = build_opportunities(S, tm)
        gwd, st, act = [], [], []
        for si, (n, s) in enumerate(S.items()):
            gd = s.matches.groupby("gameweek")["kickoff_time"].min().rename("gw_date").reset_index().rename(columns={"gameweek": "gw"})
            gd["season_idx"], gd["season"] = si, n
            gwd.append(gd)
            if not s.pgw.empty:
                keep = ["player_code", "gw", "status", "chance_of_playing_next_round", "news", "event_points", "minutes",
                        "bonus", "ep_next", "ep_this", "total_points", "goals_scored", "assists", "clean_sheets",
                        "goals_conceded", "saves", "yellow_cards", "red_cards", "own_goals", "defensive_contribution",
                        "penalties_missed", "penalties_saved", "penalties_order"]
                g = s.pgw[[c for c in keep if c in s.pgw]].copy()
                g = g.rename(columns={"chance_of_playing_next_round": "chance"})
                g["season_idx"], g["season"] = si, n
                g["gidx"] = si * 100 + g["gw"]
                st.append(g)
        gw_date = pd.concat(gwd, ignore_index=True)
        status = pd.concat(st, ignore_index=True) if st else pd.DataFrame()
        status = status.merge(gw_date[["season_idx", "gw", "gw_date"]], on=["season_idx", "gw"], how="left")
        actual = status.drop(columns=["status", "chance", "news", "total_points", "gidx", "gw_date"], errors="ignore")
        state = add_state(opp)
        fin = opp[opp["season"] != names[0]] if len(names) > 1 else opp
        per = fin.groupby(["match_id", "team"]).agg(n60=("minutes", lambda x: (x >= 60).sum()),
                                                   nplay=("minutes", lambda x: (x > 0).sum()))
        norm = (float(per["n60"].mean()), float(per["nplay"].mean()))
        return cls(S, tm, opp, status, gw_date, actual, state, norm, names)

    # ------------------------------------------------------------------
    def predict(self, asof: pd.Timestamp, season_name: str, target_gws: list[int], params: Params, rules: Rules,
                status_now: pd.DataFrame, bonus: BonusModel | None = None, fixtures: pd.DataFrame | None = None,
                team_now: pd.Series | None = None, overrides: dict | None = None, avail_model: AvailabilityModel | None = None,
                promoted_prior: dict | None = None) -> pd.DataFrame:
        """Per player x fixture expected points for `target_gws` using only information before `asof`.

        status_now: player_code, status, chance, news   (state at `asof`)
        fixtures:   match_id, gw, date, team, opp, is_home  (defaults to the season's unfinished/target matches)
        team_now:   Series player_code -> team_code for the target period
        """
        si = self.names.index(season_name)
        s = self.seasons[season_name]
        tm = self.tm
        idx = tm[tm["date"] < asof].groupby("team").size().to_dict()

        teams = sorted(set(s.teams["code"]))
        tmodel = fit_team_model(tm, asof, params, si, teams, promoted_prior)

        # positions: this season's roster, falling back to earlier seasons
        pos = s.players.set_index("player_code")["position"]
        for n in reversed(self.names[:si]):
            pos = pos.combine_first(self.seasons[n].players.set_index("player_code")["position"])
        pos = pos.dropna()
        if team_now is not None:
            pos = pos[pos.index.isin(team_now.index)]
        prof = player_profiles(self.opp, asof, params, idx, si, pos).table
        prof["team_code"] = team_now.reindex(prof.index) if team_now is not None else s.players.set_index("player_code")["team_code"].reindex(prof.index)
        prof = prof[prof["team_code"].notna()].copy()
        prof["team_code"] = prof["team_code"].astype(int)
        # learned minutes probabilities (given available), replacing the hand-built estimate
        mm = MinutesModel.fit(self.state, asof)
        cnt = tm[(tm["season"] == season_name) & (tm["date"] < asof)].groupby("team").size()
        early = (prof["team_code"].map(cnt).fillna(0) + 1 <= 4).astype(float)
        stt = latest_state(self.state, asof).reindex(prof.index)
        X = features_from_state(stt, prof["position"], early, mm.prior60)
        pr = mm.predict(X)
        prof["p60_hist"], prof["pS_hist"] = prof["p60"], prof["pS"]
        prof["p60"], prof["pS"] = pr["p60"], pr["pS"]
        if overrides:
            for pc, o in overrides.items():
                pc = int(pc)
                if pc in prof.index:
                    for k, v in o.items():
                        if k in prof.columns:
                            prof.loc[pc, k] = v

        # penalties: team rate and the evidence-weighted taker for each club
        fo = status_now.drop_duplicates("player_code").set_index("player_code").get("penalties_order")
        fo = pd.Series(dtype=float) if fo is None else pd.to_numeric(fo, errors="coerce")
        rate, league, conv = team_rates(self.opp, asof, si, idx)
        tw = taker_weights(self.opp, asof, si, idx, prof["team_code"], fo, prof["xg_share"], params.shrink.prior_season_weight)
        prof = prof.join(tw[["pen_w", "pen_share", "pen_ev", "pen_n_club", "pen_n_prev", "pen_last_gw", "fpl_order", "mover", "club_matches"]])
        prof["pen_rate"] = prof["team_code"].map(rate).fillna(league)
        prof["pen_conv"] = conv
        self.last_pen = (tw, rate, league, conv)

        if fixtures is None:
            f = s.matches[s.matches["gameweek"].isin(target_gws)]
            fixtures = pd.concat([
                f.assign(team=f["home_code"], opp=f["away_code"], is_home=1),
                f.assign(team=f["away_code"], opp=f["home_code"], is_home=0)])[["match_id", "gameweek", "kickoff_time", "team", "opp", "is_home"]]
            fixtures = fixtures.rename(columns={"gameweek": "gw", "kickoff_time": "date"})
        ctx = fixture_context(fixtures, tmodel)

        # availability curve per player over the target gameweeks
        gw_dates = [fixtures.loc[fixtures["gw"] == g, "date"].min() for g in target_gws]
        am = avail_model
        if am is None:
            max_gidx = si * 100 + (min(target_gws) - 1)
            am = AvailabilityModel.fit(self.status, max_gidx)
        rows = []
        st = status_now.drop_duplicates("player_code").set_index("player_code")
        for pc in prof.index:
            if pc in st.index:
                r = st.loc[pc]
                cur = am.curve(r["status"], r.get("chance"), r.get("news"), gw_dates, asof)
            else:
                cur = np.ones(len(target_gws))
            rows.append(pd.DataFrame({"player_code": pc, "gw": target_gws, "avail": cur}))
        avail = pd.concat(rows, ignore_index=True)
        d = expected_points(ctx, prof, avail, rules, params, bonus, norm=self.norm)
        d["asof"] = asof
        return d
