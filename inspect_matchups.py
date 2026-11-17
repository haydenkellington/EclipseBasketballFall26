
import pandas as pd
import tarfile
from pathlib import Path


# ============================================================
# STEP 1: LOAD THE FOUR MATCHUP ARCHIVES
# ============================================================

matchup_files = [
    "matchups_2024.tar.xz",
    "matchups_2025 (1).tar.xz",
    "matchups_po_2024.tar.xz",
    "matchups_po_2025.tar.xz"
]

all_matchups = []

for file_name in matchup_files:

    archive_path = Path(file_name)

    print("\n" + "=" * 60)
    print("Reading:", file_name)

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_name}"
        )

    with tarfile.open(archive_path, "r:xz") as archive:

        csv_files = [
            member for member in archive.getmembers()
            if member.isfile()
            and member.name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                f"No CSV file found inside {file_name}"
            )

        for member in csv_files:

            with archive.extractfile(member) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    low_memory=False
                )

            df["SOURCE_ARCHIVE"] = file_name
            all_matchups.append(df)

            print("CSV:", member.name)
            print("Dataset shape:", df.shape)


# Combine all four archives
matchups_combined = pd.concat(
    all_matchups,
    ignore_index=True,
    sort=False
)

print("\n" + "=" * 60)
print("COMBINED MATCHUP DATASET")
print("Shape:", matchups_combined.shape)
print("Columns:", list(matchups_combined.columns))


# ============================================================
# STEP 2: LOAD HAYDEN'S CLEANED SHOT DATASET
# ============================================================

shots_parquet = Path(
    "data/processed/nba_shots_clean.parquet"
)

shots_csv = Path(
    "data/processed/nba_shots_clean.csv.gz"
)

if shots_parquet.exists():

    try:
        shots = pd.read_parquet(shots_parquet)
        print("\nLoaded Hayden's Parquet dataset.")

    except ImportError:

        if shots_csv.exists():
            shots = pd.read_csv(shots_csv)
            print("Loaded compressed CSV instead.")
        else:
            raise

elif shots_csv.exists():

    shots = pd.read_csv(shots_csv)
    print("\nLoaded Hayden's compressed CSV dataset.")

else:
    raise FileNotFoundError(
        "Could not find Hayden's cleaned shot dataset."
    )

print("Shot dataset shape:", shots.shape)


# ============================================================
# STEP 3: STANDARDIZE IDENTIFIERS AND DATES
# ============================================================

def normalize_id(series, pad_game_id=False):

    result = (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )

    if pad_game_id:
        result = result.str.zfill(10)

    return result


# Game IDs
shots["game_id"] = normalize_id(
    shots["game_id"],
    pad_game_id=True
)

matchups_combined["game_id"] = normalize_id(
    matchups_combined["game_id"],
    pad_game_id=True
)

# Player IDs
shots["player_id"] = normalize_id(
    shots["player_id"]
)

matchups_combined["person_id"] = normalize_id(
    matchups_combined["person_id"]
)

matchups_combined["matchups_person_id"] = normalize_id(
    matchups_combined["matchups_person_id"]
)

# Team IDs in the shot dataset
for column in ["team_id", "opponent_team_id"]:

    shots[column] = normalize_id(
        shots[column]
    )

# Team IDs in the matchup dataset
for column in [
    "team_id",
    "home_team_id",
    "away_team_id"
]:

    matchups_combined[column] = normalize_id(
        matchups_combined[column]
    )

# Dates
shots["game_date"] = pd.to_datetime(
    shots["game_date"],
    errors="coerce"
)


# ============================================================
# STEP 4: VERIFY MATCHUP TEAM IDENTIFIERS
# ============================================================

# Every matchup row should belong to either the home
# team or the away team.

valid_team = (
    matchups_combined["team_id"].eq(
        matchups_combined["home_team_id"]
    )
    |
    matchups_combined["team_id"].eq(
        matchups_combined["away_team_id"]
    )
)

print("\nInvalid matchup team rows:", (~valid_team).sum())

# Identify the opposing team for each matchup record.
# The offensive player belongs to team_id.
# The defender belongs to opponent_team_id.

matchups_combined["opponent_team_id"] = (
    matchups_combined["away_team_id"].where(
        matchups_combined["team_id"].eq(
            matchups_combined["home_team_id"]
        ),
        matchups_combined["home_team_id"]
    )
)


# ============================================================
# STEP 5: VERIFY SHOT AND MATCHUP DATASET OVERLAP
# ============================================================

shot_pairs = (
    shots[
        [
            "game_id",
            "player_id",
            "team_id",
            "opponent_team_id"
        ]
    ]
    .dropna()
    .drop_duplicates()
)

matchup_pairs = (
    matchups_combined.loc[
        valid_team,
        [
            "game_id",
            "person_id",
            "team_id",
            "opponent_team_id"
        ]
    ]
    .dropna()
    .drop_duplicates()
)

matchup_pairs = matchup_pairs.rename(
    columns={
        "person_id": "player_id"
    }
)

overlap = shot_pairs.merge(
    matchup_pairs,
    on=[
        "game_id",
        "player_id",
        "team_id",
        "opponent_team_id"
    ],
    how="inner",
    validate="one_to_one"
)

print("\n" + "=" * 60)
print("TEAM-VERIFIED MATCH RESULTS")

print("Unique shot combinations:", len(shot_pairs))
print("Unique matchup combinations:", len(matchup_pairs))
print("Matching combinations:", len(overlap))

if len(shot_pairs) > 0:

    coverage = (
        len(overlap) / len(shot_pairs)
    ) * 100

    print("Coverage:", round(coverage, 2), "%")


# Find unmatched shot combinations
unmatched = shot_pairs.merge(
    matchup_pairs,
    on=[
        "game_id",
        "player_id",
        "team_id",
        "opponent_team_id"
    ],
    how="left",
    indicator=True
)

unmatched = unmatched[
    unmatched["_merge"] == "left_only"
]

print("\nUnmatched shot combinations:", len(unmatched))
print(unmatched.head(10).to_string(index=False))


# ============================================================
# STEP 6: COUNT DEFENDERS PER SHOOTER PER GAME
# ============================================================

defender_counts = (
    matchups_combined
    .dropna(
        subset=[
            "game_id",
            "team_id",
            "person_id",
            "matchups_person_id"
        ]
    )
    .groupby(
        ["game_id", "team_id", "person_id"]
    )
    .agg(
        defender_count=(
            "matchups_person_id",
            "nunique"
        ),
        matchup_rows=(
            "matchups_person_id",
            "size"
        )
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("DEFENDER COUNT ANALYSIS")

print("Shooter-game combinations:", len(defender_counts))

print("\nDefenders per shooter-game:")
print(
    defender_counts["defender_count"].describe()
)

print("\nFrequency of each defender count:")
print(
    defender_counts["defender_count"]
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 7: VALIDATE MATCHUP SHOOTING STATISTICS
# ============================================================

# Count actual shot attempts and makes in Hayden's dataset.

shots["shot_made_flag"] = pd.to_numeric(
    shots["shot_made_flag"],
    errors="coerce"
)

shot_stats = (
    shots
    .groupby(
        ["game_id", "player_id", "team_id"]
    )
    .agg(
        shot_attempts=("shot_made_flag", "size"),
        shots_made=("shot_made_flag", "sum")
    )
    .reset_index()
)

# Sum matchup shooting statistics across defenders.

matchup_stats = (
    matchups_combined
    .groupby(
        ["game_id", "person_id", "team_id"]
    )
    .agg(
        matchup_attempts=(
            "matchup_field_goals_attempted",
            "sum"
        ),
        matchup_makes=(
            "matchup_field_goals_made",
            "sum"
        ),
        defender_count=(
            "matchups_person_id",
            "nunique"
        )
    )
    .reset_index()
)

matchup_stats = matchup_stats.rename(
    columns={
        "person_id": "player_id"
    }
)

comparison = shot_stats.merge(
    matchup_stats,
    on=[
        "game_id",
        "player_id",
        "team_id"
    ],
    how="left"
)

comparison["attempt_difference"] = (
    comparison["matchup_attempts"]
    - comparison["shot_attempts"]
)

comparison["make_difference"] = (
    comparison["matchup_makes"]
    - comparison["shots_made"]
)

print("\n" + "=" * 60)
print("MATCHUP SHOOTING STATISTICS VALIDATION")

print("Shooter-game combinations:", len(comparison))

print(
    "Exact attempt-count matches:",
    comparison["attempt_difference"].eq(0).sum()
)

print(
    "Exact make-count matches:",
    comparison["make_difference"].eq(0).sum()
)

print("\nAverage attempt difference:")
print(comparison["attempt_difference"].mean())

print("\nFirst 10 mismatches:")

print(
    comparison[
        comparison["attempt_difference"].ne(0)
        | comparison["make_difference"].ne(0)
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# STEP 8: CREATE DEFENSIVE TEAM-GAME SUMMARY
# ============================================================

# Get the game date from Hayden's shot dataset.

game_dates = (
    shots
    .groupby("game_id", as_index=False)
    .agg(
        game_date=("game_date", "first")
    )
)

# Attach game dates to the matchup data.

matchups_dated = matchups_combined.merge(
    game_dates,
    on="game_id",
    how="left",
    validate="many_to_one"
)

print("\n" + "=" * 60)
print("DEFENSIVE TEAM-GAME SUMMARY")

print(
    "Matchup rows without a game date:",
    matchups_dated["game_date"].isna().sum()
)

# Convert tracking statistics to numeric values.

numeric_columns = [
    "matchup_field_goals_made",
    "matchup_field_goals_attempted",
    "matchup_three_pointers_made",
    "matchup_three_pointers_attempted",
    "matchup_blocks",
    "partial_possessions",
    "matchup_minutes_sort"
]

for column in numeric_columns:

    matchups_dated[column] = pd.to_numeric(
        matchups_dated[column],
        errors="coerce"
    )

# Aggregate by game and defending team.

defense_game = (
    matchups_dated
    .dropna(
        subset=[
            "game_date",
            "opponent_team_id"
        ]
    )
    .groupby(
        [
            "game_id",
            "game_date",
            "opponent_team_id"
        ]
    )
    .agg(
        tracked_fg_makes=(
            "matchup_field_goals_made",
            "sum"
        ),
        tracked_fg_attempts=(
            "matchup_field_goals_attempted",
            "sum"
        ),
        tracked_three_makes=(
            "matchup_three_pointers_made",
            "sum"
        ),
        tracked_three_attempts=(
            "matchup_three_pointers_attempted",
            "sum"
        ),
        tracked_blocks=(
            "matchup_blocks",
            "sum"
        ),
        tracked_partial_possessions=(
            "partial_possessions",
            "sum"
        ),
        tracked_matchup_minutes=(
            "matchup_minutes_sort",
            "sum"
        ),
        defender_count=(
            "person_id",
            "nunique"
        )
    )
    .reset_index()
)

# Calculate tracking-based shooting percentages.

defense_game["tracked_fg_pct"] = (
    defense_game["tracked_fg_makes"]
    / defense_game["tracked_fg_attempts"].where(
        defense_game["tracked_fg_attempts"] > 0
    )
)

defense_game["tracked_three_pct"] = (
    defense_game["tracked_three_makes"]
    / defense_game["tracked_three_attempts"].where(
        defense_game["tracked_three_attempts"] > 0
    )
)

print("\nSummary table shape:", defense_game.shape)

print("\nFirst 10 rows:")
print(
    defense_game.head(10).to_string(index=False)
)

print("\nNumber of defending teams:")
print(
    defense_game["opponent_team_id"].nunique()
)

print("\nNumber of games:")
print(
    defense_game["game_id"].nunique()
)

duplicate_team_games = defense_game.duplicated(
    ["game_id", "opponent_team_id"]
).sum()

print(
    "\nDuplicate team-game combinations:",
    duplicate_team_games
)

assert duplicate_team_games == 0


# ============================================================
# STEP 9: PREPARE HISTORICAL DEFENSIVE DATA
# ============================================================

# A team may appear once per game.
# We aggregate by defending team and game date.

defense_daily = (
    defense_game
    .groupby(
        [
            "opponent_team_id",
            "game_date"
        ],
        as_index=False
    )
    .agg(
        daily_fg_makes=(
            "tracked_fg_makes",
            "sum"
        ),
        daily_fg_attempts=(
            "tracked_fg_attempts",
            "sum"
        ),
        daily_three_makes=(
            "tracked_three_makes",
            "sum"
        ),
        daily_three_attempts=(
            "tracked_three_attempts",
            "sum"
        ),
        daily_blocks=(
            "tracked_blocks",
            "sum"
        ),
        daily_games=(
            "game_id",
            "nunique"
        )
    )
    .sort_values(
        [
            "opponent_team_id",
            "game_date"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# STEP 10: CALCULATE HISTORICAL LEAGUE AVERAGES
# ============================================================

league_daily = (
    defense_daily
    .groupby(
        "game_date",
        as_index=False
    )
    .agg(
        league_fg_makes=(
            "daily_fg_makes",
            "sum"
        ),
        league_fg_attempts=(
            "daily_fg_attempts",
            "sum"
        ),
        league_three_makes=(
            "daily_three_makes",
            "sum"
        ),
        league_three_attempts=(
            "daily_three_attempts",
            "sum"
        )
    )
    .sort_values("game_date")
    .reset_index(drop=True)
)

league_metrics = [
    "league_fg_makes",
    "league_fg_attempts",
    "league_three_makes",
    "league_three_attempts"
]

# Calculate cumulative statistics using only earlier dates.

for column in league_metrics:

    league_daily["prior_" + column] = (
        league_daily[column].cumsum()
        - league_daily[column]
    )

# Calculate league FG% and three-point percentage.

league_daily["league_prior_fg_pct"] = (
    league_daily["prior_league_fg_makes"]
    / league_daily["prior_league_fg_attempts"].where(
        league_daily["prior_league_fg_attempts"] > 0
    )
).fillna(0.50)

league_daily["league_prior_three_pct"] = (
    league_daily["prior_league_three_makes"]
    / league_daily["prior_league_three_attempts"].where(
        league_daily["prior_league_three_attempts"] > 0
    )
).fillna(0.35)

# Attach earlier league averages to each team-date.

defense_daily = defense_daily.merge(
    league_daily[
        [
            "game_date",
            "league_prior_fg_pct",
            "league_prior_three_pct"
        ]
    ],
    on="game_date",
    how="left",
    validate="many_to_one"
)


# ============================================================
# STEP 11: CALCULATE HISTORICAL TEAM DEFENSE STATISTICS
# ============================================================

defense_metrics = [
    "daily_fg_makes",
    "daily_fg_attempts",
    "daily_three_makes",
    "daily_three_attempts",
    "daily_blocks",
    "daily_games"
]

# Cumulative totals minus today's totals.
# This ensures that today's game is never included.

for column in defense_metrics:

    defense_daily["prior_" + column] = (
        defense_daily
        .groupby("opponent_team_id")[column]
        .cumsum()
        - defense_daily[column]
    )


# ============================================================
# STEP 12: CREATE SMOOTHED DEFENSIVE FEATURES
# ============================================================

# We use 50 prior attempts for smoothing,
# following Hayden's initial feature-engineering approach.

ALPHA = 50.0

# Historical field-goal percentage allowed.

defense_daily["matchup_defense_prior_fg_pct_smoothed"] = (
    defense_daily["prior_daily_fg_makes"]
    + ALPHA * defense_daily["league_prior_fg_pct"]
) / (
    defense_daily["prior_daily_fg_attempts"]
    + ALPHA
)

# Historical three-point percentage allowed.

defense_daily["matchup_defense_prior_three_pct_smoothed"] = (
    defense_daily["prior_daily_three_makes"]
    + ALPHA * defense_daily["league_prior_three_pct"]
) / (
    defense_daily["prior_daily_three_attempts"]
    + ALPHA
)

# Historical attempt counts.

defense_daily["matchup_defense_prior_fg_attempts"] = (
    defense_daily["prior_daily_fg_attempts"]
)

defense_daily["matchup_defense_prior_three_attempts"] = (
    defense_daily["prior_daily_three_attempts"]
)

# Historical games played in the matchup dataset.

defense_daily["matchup_defense_prior_games"] = (
    defense_daily["prior_daily_games"]
)

# Indicator showing whether earlier matchup history exists.

defense_daily["matchup_defense_has_history"] = (
    defense_daily["prior_daily_games"]
    .gt(0)
    .astype("int8")
)


# ============================================================
# STEP 13: CHECK FOR TEMPORAL DATA LEAKAGE
# ============================================================

# The history for a team-date must end strictly
# before the current game date.

defense_daily["history_through"] = (
    defense_daily
    .groupby("opponent_team_id")["game_date"]
    .shift(1)
)

has_previous_history = (
    defense_daily["history_through"].notna()
)

assert (
    defense_daily.loc[
        has_previous_history,
        "history_through"
    ]
    <
    defense_daily.loc[
        has_previous_history,
        "game_date"
    ]
).all()

print("\n" + "=" * 60)
print("HISTORICAL DATA LEAKAGE CHECK: PASSED")


# ============================================================
# STEP 14: PREPARE THE DEFENSIVE FEATURE TABLE
# ============================================================

matchup_features = [
    "matchup_defense_prior_fg_pct_smoothed",
    "matchup_defense_prior_three_pct_smoothed",
    "matchup_defense_prior_fg_attempts",
    "matchup_defense_prior_three_attempts",
    "matchup_defense_prior_games",
    "matchup_defense_has_history"
]

defense_feature_table = defense_daily[
    [
        "opponent_team_id",
        "game_date"
    ] + matchup_features
].copy()

# There must be only one record per defending team and date.

assert not defense_feature_table.duplicated(
    [
        "opponent_team_id",
        "game_date"
    ]
).any()


# ============================================================
# STEP 15: ADD HISTORICAL DEFENSIVE FEATURES TO EACH SHOT
# ============================================================

shots_with_matchups = shots.merge(
    defense_feature_table,
    on=[
        "opponent_team_id",
        "game_date"
    ],
    how="left",
    validate="many_to_one"
)


# ============================================================
# STEP 16: VALIDATE THE FINAL MERGE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MERGE VALIDATION")

print("Original shot rows:", len(shots))
print("Merged shot rows:", len(shots_with_matchups))

# The merge must not create or remove shot rows.

assert (
    len(shots_with_matchups)
    ==
    len(shots)
)

print("Shot row count check: PASSED")

# Check that individual shots have not been duplicated.

if "game_event_id" in shots_with_matchups.columns:

    duplicate_shots = shots_with_matchups.duplicated(
        [
            "game_id",
            "game_event_id"
        ]
    ).sum()

    print("Duplicate shot records:", duplicate_shots)

    assert duplicate_shots == 0

    print("Shot uniqueness check: PASSED")


# Check missing historical defensive features.

missing_features = (
    shots_with_matchups[matchup_features]
    .isna()
    .sum()
)

print("\nMissing values in new features:")
print(missing_features)

print(
    "\nShots without a matching defensive history record:",
    shots_with_matchups[
        "matchup_defense_prior_games"
    ].isna().sum()
)


# Check that percentages are valid.

rate_columns = [
    "matchup_defense_prior_fg_pct_smoothed",
    "matchup_defense_prior_three_pct_smoothed"
]

for column in rate_columns:

    valid_rates = shots_with_matchups[column].dropna()

    assert valid_rates.between(0, 1).all()

print("\nDefensive percentage range check: PASSED")


# ============================================================
# STEP 17: DISPLAY THE NEW FEATURES
# ============================================================

print("\n" + "=" * 60)
print("NEW HISTORICAL DEFENSIVE FEATURES")

print(
    shots_with_matchups[matchup_features]
    .describe()
    .round(4)
)

print("\nExample rows:")

display_columns = [
    "game_id",
    "game_date",
    "player_name",
    "opponent_team_id"
] + matchup_features

print(
    shots_with_matchups[display_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# STEP 18: SAVE THE NEW DATASET
# ============================================================

output_directory = Path(
    "data/processed"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_directory
    / "nba_shots_with_matchup_features.parquet"
)

shots_with_matchups.to_parquet(
    output_path,
    index=False
)

# Also save the team-game summary for inspection.

summary_path = (
    output_directory
    / "matchup_defense_team_game.csv"
)

defense_game.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 60)
print("PROCESS COMPLETE")

print("New shot dataset saved to:")
print(output_path)

print("\nTeam-game summary saved to:")
print(summary_path)

print("\nOriginal Hayden dataset was not overwritten.")
print("Total shot records:", len(shots_with_matchups))
print("New defensive features:", len(matchup_features))

print("\nAll processing steps completed.")
