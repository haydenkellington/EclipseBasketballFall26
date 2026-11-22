"""Recover the 189 missing matchup-derived features for game 0022400856.

Run from any working directory with the Python interpreter used for the
project packages. The script reads the original integrated shot dataset and
team-game matchup summary, recovers the affected pregame team features using
only matchup games strictly before 2025-02-28, validates the result, and writes
an untouched-source-safe v2 Parquet file.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "nba_shots_with_matchup_features.parquet"
MATCHUP_PATH = DATA_DIR / "matchup_defense_team_game.csv"
OUTPUT_PATH = DATA_DIR / "nba_shots_with_matchup_features_v2.parquet"

TARGET_GAME_ID = "0022400856"
TARGET_DATE = pd.Timestamp("2025-02-28")
ALPHA = 50.0

NUMERIC_FEATURES = [
    "matchup_defense_prior_fg_pct_smoothed",
    "matchup_defense_prior_three_pct_smoothed",
    "matchup_defense_prior_fg_attempts",
    "matchup_defense_prior_three_attempts",
    "matchup_defense_prior_games",
]
BOOLEAN_FEATURE = "matchup_defense_has_history"
ALL_FEATURES = NUMERIC_FEATURES + [BOOLEAN_FEATURE]


def to_nullable_bool(value):
    """Convert bool, 0/1, or common string values into bool / pd.NA."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
        raise ValueError(f"Cannot convert numeric value {value!r} to Boolean.")

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False

    raise ValueError(f"Unrecognized Boolean value: {value!r}")


def main():
    print("=" * 72)
    print("RECOVER MISSING MATCHUP FEATURES")
    print("=" * 72)
    print("Python executable:", sys.executable)
    print("Project root:", ROOT)

    # --------------------------------------------------------
    # 2. CHECK FILES AND LOAD DATA
    # --------------------------------------------------------
    for path in (INPUT_PATH, MATCHUP_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    shots = pd.read_parquet(INPUT_PATH)
    defense_game = pd.read_csv(
        MATCHUP_PATH,
        dtype={"game_id": "string", "opponent_team_id": "string"},
    )

    shots.columns = shots.columns.str.strip().str.lower()
    defense_game.columns = defense_game.columns.str.strip().str.lower()

    print("\nInput shot dataset:", shots.shape)
    print("Team-game matchup summary:", defense_game.shape)

    required_shot_columns = [
        "game_id", "game_date", "opponent_team_id", "shot_made_flag", *ALL_FEATURES
    ]
    missing_shot_cols = [c for c in required_shot_columns if c not in shots.columns]
    if missing_shot_cols:
        raise ValueError(f"Shot dataset is missing required columns: {missing_shot_cols}")

    required_matchup_cols = [
        "game_id", "game_date", "opponent_team_id", "tracked_fg_makes",
        "tracked_fg_attempts", "tracked_three_makes", "tracked_three_attempts",
    ]
    missing_matchup_cols = [c for c in required_matchup_cols if c not in defense_game.columns]
    if missing_matchup_cols:
        raise ValueError(f"Matchup summary is missing required columns: {missing_matchup_cols}")

    # --------------------------------------------------------
    # 3. NORMALIZE IDENTIFIERS AND DATES
    # --------------------------------------------------------
    shots["game_id"] = shots["game_id"].astype("string").str.zfill(10)
    shots["opponent_team_id"] = shots["opponent_team_id"].astype("string")
    shots["game_date"] = pd.to_datetime(shots["game_date"], errors="coerce")

    defense_game["game_id"] = defense_game["game_id"].astype("string").str.zfill(10)
    defense_game["opponent_team_id"] = defense_game["opponent_team_id"].astype("string")
    defense_game["game_date"] = pd.to_datetime(defense_game["game_date"], errors="coerce")

    stat_cols = [
        "tracked_fg_makes", "tracked_fg_attempts",
        "tracked_three_makes", "tracked_three_attempts",
    ]
    for col in stat_cols:
        defense_game[col] = pd.to_numeric(defense_game[col], errors="coerce")

    if defense_game[stat_cols].isna().any().any():
        raise ValueError("Matchup summary contains missing/non-numeric tracked statistics.")

    # --------------------------------------------------------
    # 4. IDENTIFY THE AFFECTED SHOTS
    # --------------------------------------------------------
    print("\n--- IDENTIFY MISSING FEATURES ---")

    missing_any = shots[ALL_FEATURES].isna().any(axis=1)
    missing_all = shots[ALL_FEATURES].isna().all(axis=1)
    partial_missing = missing_any & ~missing_all

    print("Rows with all six features missing:", int(missing_all.sum()))
    print("Rows with only some features missing:", int(partial_missing.sum()))

    if partial_missing.any():
        raise ValueError(
            "Some rows have only a subset of matchup features missing. "
            "Investigate those rows before recovery."
        )

    affected = shots.loc[missing_all].copy()
    affected_games = affected["game_id"].dropna().unique().tolist()

    print("Affected game IDs:", affected_games)

    if len(affected_games) == 0:
        raise ValueError("No missing matchup-feature rows were found; nothing to recover.")
    if affected_games != [TARGET_GAME_ID]:
        raise ValueError(
            f"Expected only game {TARGET_GAME_ID}, but found {affected_games}. "
            "Stop and inspect the missing rows."
        )

    affected_counts = affected["opponent_team_id"].value_counts(dropna=False)
    print("\nAffected shots by defending team:")
    print(affected_counts.to_string())

    teams = affected["opponent_team_id"].dropna().unique().tolist()
    if len(teams) != 2:
        raise ValueError(f"Expected two defending teams in affected game; found {teams}")

    # --------------------------------------------------------
    # 5. HISTORICAL MATCHUP RECORDS STRICTLY BEFORE TARGET DATE
    # --------------------------------------------------------
    print("\n--- CALCULATE HISTORICAL DEFENSIVE FEATURES ---")

    historical = defense_game.loc[
        defense_game["game_date"] < TARGET_DATE
    ].copy()

    if historical.empty:
        raise ValueError("No matchup history exists before the affected game date.")

    # League prior rates from all team-game defensive records before target date.
    league_fg_attempts = historical["tracked_fg_attempts"].sum()
    league_three_attempts = historical["tracked_three_attempts"].sum()

    if league_fg_attempts <= 0 or league_three_attempts <= 0:
        raise ValueError("League historical attempt totals must be positive.")

    league_fg_rate = historical["tracked_fg_makes"].sum() / league_fg_attempts
    league_three_rate = historical["tracked_three_makes"].sum() / league_three_attempts

    print(f"Historical league FG%: {league_fg_rate:.4f}")
    print(f"Historical league 3P%: {league_three_rate:.4f}")

    team_features = {}

    for team in teams:
        team_history = historical.loc[
            historical["opponent_team_id"] == team
        ].copy()

        if team_history.empty:
            raise ValueError(f"No prior matchup records found for defending team {team}.")

        fg_makes = team_history["tracked_fg_makes"].sum()
        fg_attempts = team_history["tracked_fg_attempts"].sum()
        three_makes = team_history["tracked_three_makes"].sum()
        three_attempts = team_history["tracked_three_attempts"].sum()
        prior_games = team_history["game_id"].nunique()

        if fg_attempts <= 0 or three_attempts <= 0:
            raise ValueError(f"Invalid historical attempt totals for team {team}.")

        features = {
            "matchup_defense_prior_fg_pct_smoothed": (
                fg_makes + ALPHA * league_fg_rate
            ) / (fg_attempts + ALPHA),
            "matchup_defense_prior_three_pct_smoothed": (
                three_makes + ALPHA * league_three_rate
            ) / (three_attempts + ALPHA),
            "matchup_defense_prior_fg_attempts": float(fg_attempts),
            "matchup_defense_prior_three_attempts": float(three_attempts),
            "matchup_defense_prior_games": int(prior_games),
            BOOLEAN_FEATURE: True,
        }

        team_features[team] = features

        print(f"\nDefending team: {team}")
        print("Previous games:", prior_games)
        print(
            "Smoothed FG%:",
            f"{features['matchup_defense_prior_fg_pct_smoothed']:.2%}",
        )
        print(
            "Smoothed 3P%:",
            f"{features['matchup_defense_prior_three_pct_smoothed']:.2%}",
        )
        print("Prior FG attempts:", int(fg_attempts))
        print("Prior 3P attempts:", int(three_attempts))

    # --------------------------------------------------------
    # 6. ASSIGN THE FIVE NUMERIC FEATURES
    # --------------------------------------------------------
    print("\n--- INSERT RECOVERED FEATURES ---")

    for column in NUMERIC_FEATURES:
        # Float dtype safely accepts NaN during assignment.
        shots[column] = pd.to_numeric(shots[column], errors="coerce").astype("float64")

        for team, features in team_features.items():
            mask = missing_all & shots["opponent_team_id"].eq(team)
            shots.loc[mask, column] = float(features[column])

    # --------------------------------------------------------
    # 7. ASSIGN THE BOOLEAN FEATURE SAFELY
    # --------------------------------------------------------
    # The source Parquet may have represented this column as float (0.0/1.0)
    # because it originally contained missing values. Convert each scalar
    # explicitly to bool or pd.NA before using Pandas' nullable Boolean dtype.
    history_bool = shots[BOOLEAN_FEATURE].map(to_nullable_bool).astype("boolean")

    for team in teams:
        mask = missing_all & shots["opponent_team_id"].eq(team)
        history_bool.loc[mask] = True

    shots[BOOLEAN_FEATURE] = history_bool

    # --------------------------------------------------------
    # 8. VALIDATE BEFORE SAVING
    # --------------------------------------------------------
    print("\n--- FINAL VALIDATION ---")

    remaining_missing = int(shots[ALL_FEATURES].isna().any(axis=1).sum())
    affected_after = shots.loc[shots["game_id"] == TARGET_GAME_ID]

    print("Rows in affected game:", len(affected_after))
    print("Rows with any missing matchup features:", remaining_missing)
    print("Original row count:", len(shots))

    print("\nRecovered feature values by defending team:")
    print(
        affected_after.groupby("opponent_team_id")[ALL_FEATURES]
        .first()
        .to_string()
    )

    if remaining_missing != 0:
        raise ValueError(
            f"Validation failed: {remaining_missing} rows still have missing matchup features."
        )

    if len(affected_after) != int(affected_counts.sum()):
        raise ValueError("Affected game row count changed unexpectedly.")

    # Validate all historical rates as proportions.
    for col in NUMERIC_FEATURES[:2]:
        if not shots[col].between(0, 1).all():
            raise ValueError(f"Invalid percentage values remain in {col}.")

    # Ensure all recovered observations have a history flag.
    recovered_flags = shots.loc[missing_all, BOOLEAN_FEATURE]
    if not recovered_flags.fillna(False).all():
        raise ValueError("Some recovered rows do not have has_history=True.")

    # --------------------------------------------------------
    # 9. SAVE A NEW DATASET, KEEPING THE SOURCE UNCHANGED
    # --------------------------------------------------------
    shots.to_parquet(OUTPUT_PATH, index=False)

    # Read back and verify the saved artifact.
    saved = pd.read_parquet(OUTPUT_PATH)
    if len(saved) != len(shots):
        raise ValueError("Saved Parquet row count does not match the in-memory dataset.")
    if saved[ALL_FEATURES].isna().any().any():
        raise ValueError("Saved Parquet still contains missing matchup features.")

    print("\n" + "=" * 72)
    print("RECOVERY COMPLETE")
    print("=" * 72)
    print("Saved corrected dataset:", OUTPUT_PATH)
    print("Final shape:", saved.shape)
    print("All six matchup features complete: True")


if __name__ == "__main__":
    main()
