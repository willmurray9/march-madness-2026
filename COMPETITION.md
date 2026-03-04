# March Machine Learning Mania 2026

## Overview

Forecast the outcomes of both the men's and women's 2026 collegiate basketball tournaments by submitting predictions for every possible tournament matchup.

***

## Description

In this twelfth annual March Machine Learning Mania competition, participants predict the outcomes of this year's college basketball tournaments using historical data and computing power. You are provided historical NCAA game data to forecast outcomes for Division 1 Men's and Women's basketball tournaments.

Predictions are made for **every possible matchup** in the tournament, evaluated using the **Brier score**. Prior to the start of the tournaments, the leaderboard reflects scores from 2021–2025 only. Kaggle will periodically rescore once 2026 games begin.

***

## Evaluation

Submissions are evaluated on the [Brier score](https://en.wikipedia.org/wiki/Brier_score) between predicted probabilities and actual game outcomes (equivalent to mean squared error in this context).

***

## Submission File Format

- Men's and Women's tournaments are combined into **one submission file**
- Predict hypothetical results for **every possible team matchup** (not just tournament-selected teams)
- Each game has a unique ID: `SSSS_XXXX_YYYY`
  - `SSSS` = season year
  - `XXXX` = lower TeamID
  - `YYYY` = higher TeamID
- **Predict the probability that the lower TeamID team wins**
- Men's TeamIDs: `1000–1999` | Women's TeamIDs: `3000–3999`

### Example

```csv
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.5
2026_1101_1104,0.5
```

> Your 2026 submissions will score 0.0 if submitted in the correct format. The leaderboard only becomes meaningful once 2026 tournaments begin.

***

## Dataset Description

Data files covering both men's and women's data:
- Files prefixed with `M` = men's data only
- Files prefixed with `W` = women's data only
- `Cities.csv` and `Conferences.csv` = shared between both

All files are complete through **February 4th** of the current season. Updates will be provided as the season progresses toward mid-March.

> **Season convention:** A season is identified by the year the tournament is played (e.g., "2026 season" = the tournament played in spring 2026).

***

## Data Section 1 — The Basics

Includes team IDs, tournament seeds, game results, and season details since the 1984–85 season.

### MTeams.csv / WTeams.csv

| Column | Description |
|---|---|
| `TeamID` | 4-digit unique team identifier (M: 1000–1999, W: 3000–3999) |
| `TeamName` | Compact team name (16 characters or fewer) |
| `FirstD1Season` | First season as Division-I school (men's only) |
| `LastD1Season` | Last season as Division-I school; `2026` = currently active (men's only) |

### MSeasons.csv / WSeasons.csv

| Column | Description |
|---|---|
| `Season` | Year the tournament was played |
| `DayZero` | Date corresponding to DayNum=0 for that season |
| `RegionW/X/Y/Z` | Region names assigned alphabetically and by bracket pairing |

### DayNum Reference (Men's)

| DayNum | Event |
|---|---|
| 132 | Selection Sunday / last regular season day |
| 134–135 | Play-in games |
| 136–137 | Round 1 (64 to 32) |
| 138–139 | Round 2 (32 to 16) |
| 143–144 | Sweet Sixteen (16 to 8) |
| 145–146 | Elite Eight (8 to 4) |
| 152 | Final Four (4 to 2) |
| 154 | National Championship (2 to 1) |

### MNCAATourneySeeds.csv / WNCAATourneySeeds.csv

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `Seed` | 3–4 character seed (e.g., W01, Z13a); region letter + seed number + optional play-in suffix |
| `TeamID` | Team identifier |

> Seeds and the final 68-team field will not be known until Selection Sunday, March 15, 2026 (DayNum=132).

### MRegularSeasonCompactResults.csv / WRegularSeasonCompactResults.csv

Game-by-game results for all games on DayNum 132 or earlier.

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `DayNum` | Day of game (0–132) |
| `WTeamID` | Winning team ID |
| `WScore` | Winning team score |
| `LTeamID` | Losing team ID |
| `LScore` | Losing team score |
| `WLoc` | Location of winning team: H (home), A (away), N (neutral) |
| `NumOT` | Number of overtime periods |

### MNCAATourneyCompactResults.csv / WNCAATourneyCompactResults.csv

Same format as regular season compact results, but for NCAA tournament games only. Men's games are always neutral site (WLoc=N).

### SampleSubmissionStage1.csv / SampleSubmissionStage2.csv

- **Stage 1:** All possible matchups from seasons 2022–2025 (for model development)
- **Stage 2:** All possible matchups for 2026 (for final submission)

| Column | Description |
|---|---|
| `ID` | Format: SSSS_XXXX_YYYY (season_lowerTeamID_higherTeamID) |
| `Pred` | Predicted win probability for the lower TeamID team |

***

## Data Section 2 — Team Box Scores

Detailed game-by-game team stats available from 2003 (men) and 2010 (women).

### Files
- `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv`
- `MNCAATourneyDetailedResults.csv` / `WNCAATourneyDetailedResults.csv`

First 8 columns are identical to Compact Results. Additional columns:

| Column | Description |
|---|---|
| `WFGM` / `LFGM` | Field goals made |
| `WFGA` / `LFGA` | Field goals attempted |
| `WFGM3` / `LFGM3` | Three-pointers made |
| `WFGA3` / `LFGA3` | Three-pointers attempted |
| `WFTM` / `LFTM` | Free throws made |
| `WFTA` / `LFTA` | Free throws attempted |
| `WOR` / `LOR` | Offensive rebounds |
| `WDR` / `LDR` | Defensive rebounds |
| `WAst` / `LAst` | Assists |
| `WTO` / `LTO` | Turnovers |
| `WStl` / `LStl` | Steals |
| `WBlk` / `LBlk` | Blocks |
| `WPF` / `LPF` | Personal fouls |

> **Note:** FGM = total field goals (2PT + 3PT). Two-point FGM = FGM - FGM3. Total points = (2 x FGM) + FGM3 + FTM.

***

## Data Section 3 — Geography

City-level game location data available from the 2010 season onward.

### Cities.csv

| Column | Description |
|---|---|
| `CityID` | 4-digit unique city identifier |
| `City` | City name |
| `State` | State abbreviation (non-US locations use alternate codes, e.g., MX for Mexico) |

### MGameCities.csv / WGameCities.csv

| Column | Description |
|---|---|
| `Season`, `DayNum`, `WTeamID`, `LTeamID` | Unique game identifier (4 columns) |
| `CRType` | Game type: Regular, NCAA, or Secondary |
| `CityID` | City where the game was played |

***

## Data Section 4 — Public Rankings (Men's Only)

Weekly ordinal rankings from dozens of systems (Pomeroy, Sagarin, RPI, ESPN, etc.) since the 2003 season.

### MMasseyOrdinals.csv

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `RankingDayNum` | First day the ranking is valid for predictions (0–133); final pre-tournament rankings use 133 |
| `SystemName` | 3-letter ranking system abbreviation |
| `TeamID` | Ranked team ID |
| `OrdinalRank` | Overall ranking (e.g., 1 through 351+) |

> **Warning:** Kaggle has no control over ranking release timing; not all systems may be available before the submission deadline.

***

## Data Section 5 — Supplements

### MTeamCoaches.csv

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `TeamID` | Team identifier |
| `FirstDayNum` | First day this coach was head coach |
| `LastDayNum` | Last day this coach was head coach |
| `CoachName` | Coach name in first_last lowercase format |

### Conferences.csv

| Column | Description |
|---|---|
| `ConfAbbrev` | Short conference abbreviation |
| `Description` | Full conference name |

### MTeamConferences.csv / WTeamConferences.csv

Historical conference affiliations per team per season.

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `TeamID` | Team identifier |
| `ConfAbbrev` | Conference abbreviation |

### MConferenceTourneyGames.csv / WConferenceTourneyGames.csv

Conference tournament games from 2001 (men) / 2002 (women).

| Column | Description |
|---|---|
| `ConfAbbrev` | Conference identifier |
| `Season`, `DayNum`, `WTeamID`, `LTeamID` | Unique game identifier |

### MSecondaryTourneyTeams.csv / WSecondaryTourneyTeams.csv

Teams participating in non-NCAA postseason tournaments.

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `SecondaryTourney` | Tournament abbreviation |
| `TeamID` | Participating team ID |

Men's secondary tournaments: NIT, CBI, CBC, CIT, V16 (Vegas 16), TBC (The Basketball Classic)
Women's secondary tournaments: WBI, WBIT, WNIT

### MSecondaryTourneyCompactResults.csv / WSecondaryTourneyCompactResults.csv

Same as Compact Results format plus `SecondaryTourney` column. These games occur after DayNum=132 and are NOT listed in Regular Season files.

### MTeamSpellings.csv / WTeamSpellings.csv

Alternative team name spellings for linking external data to TeamID values.

| Column | Description |
|---|---|
| `TeamNameSpelling` | Alternate spelling in all lowercase |
| `TeamID` | Corresponding team ID |

### MNCAATourneySlots.csv / WNCAATourneySlots.csv

Bracket pairing structure showing how seeds are matched across rounds.

| Column | Description |
|---|---|
| `Season` | Tournament year |
| `Slot` | Game slot ID (e.g., R1W1 for regular rounds, W16 for play-ins) |
| `StrongSeed` | Expected stronger seed or winning slot reference for Round 2+ |
| `WeakSeed` | Expected weaker seed or winning slot reference for Round 2+ |

### MNCAATourneySeedRoundSlots.csv (Men's Only)

Maps each tournament seed to its bracket slot and possible DayNums per round.

| Column | Description |
|---|---|
| `Seed` | Tournament seed |
| `GameRound` | Round number: 0=play-in, 1-2=first weekend, 3-4=second weekend, 5-6=semis and finals |
| `GameSlot` | Bracket slot for that round |
| `EarlyDayNum` | Earliest possible game day number |
| `LateDayNum` | Latest possible game day number |

> Note: The 2021 men's tournament had non-standard scheduling and did not follow traditional DayNum assignments. No equivalent file exists for women's data due to scheduling variability.
