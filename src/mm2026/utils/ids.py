from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchupID:
    season: int
    team_low: int
    team_high: int



def parse_matchup_id(matchup_id: str) -> MatchupID:
    season_s, team_a_s, team_b_s = matchup_id.split("_")
    season = int(season_s)
    team_a = int(team_a_s)
    team_b = int(team_b_s)
    low, high = (team_a, team_b) if team_a < team_b else (team_b, team_a)
    return MatchupID(season=season, team_low=low, team_high=high)



def format_matchup_id(season: int, team_low: int, team_high: int) -> str:
    if team_low > team_high:
        team_low, team_high = team_high, team_low
    return f"{season}_{team_low}_{team_high}"
