from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path.cwd()

INTEGRATED_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "nba_shots_with_matchup_features.parquet"
)

DEFENSE_GAME_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "matchup_defense_team_game.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "nba_shots_with_matchup_features_v2.parquet"
)

TARGET_GAME = "0022400856"
TARGET_DATE = pd.Timestamp("2025-02-28")

ALPHA = 50

NEW_FEATURES = [
    "matchup_defense_prior_fg_pct_smoothed",
    "matchup_defense_prior_three_pct_smoothed",
    "matchup_defense_prior_fg_attempts",
    "matchup_defense_prior_three_attempts",
    "matchup_defense_prior_games",
    "matchup_defense_has_history"
]

NUMERIC_FEATURES = [
    "matchup_defense_prior_fg_pct_smoothed",
    "matchup_defense_prior_three_pct_smoothed",
    "matchup_defense_prior_fg_attempts",
    "matchup_defense_prior_three_attempts",
    "matchup_defense_prior_games"
]

BOOLEAN_FEATURE = "matchup_defense_has_history"


# ============================================================
# 2. CHECK FILES AND LOAD DATA
# ============================================================

print("--- CHECKING FILES ---")

if not INTEGRATED_PATH.exists():
    raise FileNotFoundError(f"Missing file: {INTEGRATED_PATH}")

if not DEFENSE_GAME_PATH.exists():
    raise FileNotFoundError(f"Missing file: {DEFENSE_GAME_PATH}")

integrated = pd.read_parquet(INTEGRATED_PATH)

defense_game = pd.read_csv(
    DEFENSE_GAME_PATH,
    dtype={
        "game_id": "string",
        "opponent_team_id": "string"
    }
)

print("Integrated dataset loaded:", integrated.shape)
print("Defense game dataset loaded:", defense_game.shape)


# ============================================================
# 3. STANDARDIZE IDs, DATES AND NUMERIC COLUMNS
# ============================================================

integrated["game_id"] = integrated["game_id"].astype("string")
integrated["opponent_team_id"] = (
    integrated["opponent_team_id"].astype("string")
)

integrated["game_date"] = pd.to_datetime(
    integrated["game_date"],
    errors="coerce"
)

defense_game["game_id"] = defense_game["game_id"].astype("string")
defense_game["opponent_team_id"] = (
    defense_game["opponent_team_id"].astype("string")
)

defense_game["game_date"] = pd.to_datetime(
    defense_game["game_date"],
    errors="coerce"
)

# Check that the required columns exist
required_columns = [
    "game_id",
    "opponent_team_id",
    "game_date",
    "tracked_fg_makes",
    "tracked_fg_attempts",
    "tracked_three_makes",
    "tracked_three_attempts"
]

missing_columns = [
    column for column in required_columns
    if column not in defense_game.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in defense_game: {missing_columns}"
    )

# Convert matchup statistics to numeric
stat_columns = [
    "tracked_fg_makes",
    "tracked_fg_attempts",
    "tracked_three_makes",
    "tracked_three_attempts"
]

for column in stat_columns:
    defense_game[column] = pd.to_numeric(
        defense_game[column],
        errors="coerce"
    )


# ============================================================
# 4. IDENTIFY THE MISSING SHOT RECORDS
# ============================================================

print("\n--- IDENTIFYING MISSING FEATURES ---")

missing_any = integrated[NEW_FEATURES].isna().any(axis=1)

missing_all = integrated[NEW_FEATURES].isna().all(axis=1)

partial_missing = missing_any & ~missing_all

print("Rows with all six features missing:", missing_all.sum())
print("Rows with only some features missing:", partial_missing.sum())

if partial_missing.sum() > 0:
    raise ValueError(
        "Some rows have only partially missing matchup features. "
        "Investigate these before continuing."
    )

missing_game_ids = (
    integrated.loc[missing_all, "game_id"]
    .dropna()
    .unique()
)

print("Games with missing features:", missing_game_ids)

if len(missing_game_ids) != 1 or missing_game_ids[0] != TARGET_GAME:
    raise ValueError(
        "The missing records do not match the expected affected game. "
        "Stop and investigate before modifying the dataset."
    )

affected_rows = missing_all.copy()

print("\nMissing rows by opponent:")
print(
    integrated.loc[
        affected_rows, "opponent_team_id"
    ].value_counts()
)


# ============================================================
# 5. GET ALL HISTORICAL MATCHUP DATA BEFORE THE GAME
# ============================================================

print("\n--- CALCULATING HISTORICAL DEFENSIVE RATES ---")

# IMPORTANT:
# Only use games played BEFORE February 28, 2025.
# This prevents information from the target game or future
# games from entering the historical features.

historical = defense_game[
    defense_game["game_date"] < TARGET_DATE
].copy()

if historical.empty:
    raise ValueError("No historical matchup records available.")

# League-wide historical shooting rates
league_fg_attempts = historical[
    "tracked_fg_attempts"
].sum()

league_three_attempts = historical[
    "tracked_three_attempts"
].sum()

if league_fg_attempts <= 0 or league_three_attempts <= 0:
    raise ValueError("Invalid league-wide attempt totals.")

league_fg_rate = (
    historical["tracked_fg_makes"].sum()
    / league_fg_attempts
)

league_three_rate = (
    historical["tracked_three_makes"].sum()
    / league_three_attempts
)

print("Historical league FG%:", round(league_fg_rate, 4))
print("Historical league 3P%:", round(league_three_rate, 4))


# ============================================================
# 6. CALCULATE FEATURES FOR EACH AFFECTED DEFENDING TEAM
# ============================================================

teams = (
    integrated.loc[affected_rows, "opponent_team_id"]
    .dropna()
    .unique()
)

team_features = {}

for team in teams:

    team_history = historical[
        historical["opponent_team_id"] == team
    ].copy()

    if team_history.empty:
        raise ValueError(
            f"No historical matchup records for team {team}"
        )

    # Historical shooting totals
    fg_makes = team_history["tracked_fg_makes"].sum()
    fg_attempts = team_history["tracked_fg_attempts"].sum()

    three_makes = team_history["tracked_three_makes"].sum()
    three_attempts = team_history["tracked_three_attempts"].sum()

    # Apply the same smoothing approach used in our
    # original feature engineering
    smoothed_fg_pct = (
        fg_makes + ALPHA * league_fg_rate
    ) / (
        fg_attempts + ALPHA
    )

    smoothed_three_pct = (
        three_makes + ALPHA * league_three_rate
    ) / (
        three_attempts + ALPHA
    )

    features = {
        "matchup_defense_prior_fg_pct_smoothed":
            smoothed_fg_pct,

        "matchup_defense_prior_three_pct_smoothed":
            smoothed_three_pct,

        "matchup_defense_prior_fg_attempts":
            fg_attempts,

        "matchup_defense_prior_three_attempts":
            three_attempts,

        "matchup_defense_prior_games":
            team_history["game_id"].nunique(),

        "matchup_defense_has_history":
            True
    }

    team_features[team] = features

    print(f"\nDefending team: {team}")
    print("Previous games:", features["matchup_defense_prior_games"])
    print("Smoothed FG%:", round(smoothed_fg_pct * 100, 2), "%")
    print("Smoothed 3P%:", round(smoothed_three_pct * 100, 2), "%")
    print("Prior FG attempts:", fg_attempts)
    print("Prior 3P attempts:", three_attempts)


# ============================================================
# 7. INSERT THE RECOVERED FEATURES
# ============================================================

print("\n--- INSERTING RECOVERED FEATURES ---")

# Prepare numeric columns as float arrays so that values can
# safely be inserted even if the original columns use Arrow
# or nullable numeric data types.

for column in NUMERIC_FEATURES:

    values = pd.to_numeric(
        integrated[column],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    for team, features in team_features.items():

        mask = (
            affected_rows
            & integrated["opponent_team_id"].eq(team)
        )

        values[mask.to_numpy()] = float(features[column])

    integrated[column] = values


# Handle the Boolean column separately.
# This avoids the float64-versus-True assignment error.

history_values = (
    integrated[BOOLEAN_FEATURE]
    .astype("object")
    .to_numpy()
)

for team, features in team_features.items():

    mask = (
        affected_rows
        & integrated["opponent_team_id"].eq(team)
    )

    history_values[mask.to_numpy()] = True

integrated[BOOLEAN_FEATURE] = pd.Series(
    pd.array(history_values, dtype="boolean"),
    index=integrated.index
)


# ============================================================
# 8. VALIDATE THE CORRECTED DATASET
# ============================================================

print("\n--- FINAL VALIDATION ---")

remaining_missing = integrated[NEW_FEATURES].isna().any(axis=1).sum()

target_game_rows = integrated[
    integrated["game_id"] == TARGET_GAME
]

print("Original integrated row count:", len(integrated))
print("Rows in affected game:", len(target_game_rows))
print("Rows with missing matchup features:", remaining_missing)

print("\nCorrected feature values:")
print(
    target_game_rows.groupby("opponent_team_id")[
        NEW_FEATURES
    ].first().to_string()
)

# Confirm all six features were recovered for affected rows
if remaining_missing != 0:
    raise ValueError(
        f"Validation failed: {remaining_missing} rows still have "
        "missing matchup features."
    )

if len(target_game_rows) != 189:
    raise ValueError(
        "Unexpected number of rows in the affected game."
    )

if len(integrated) != 467537:
    raise ValueError(
        "The dataset row count changed unexpectedly."
    )

print("\nAll validation checks passed.")


# ============================================================
# 9. SAVE THE CORRECTED DATASET
# ============================================================

integrated.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\n--- SAVE COMPLETE ---")
print("Saved corrected dataset to:")
print(OUTPUT_PATH)
print("\nFinal dataset shape:", integrated.shape)
print("Final feature count:", len(integrated.columns))