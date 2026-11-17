
import numpy as np
import pandas as pd
import tarfile


# ============================================================
# 1. FUNCTION TO READ ANY NBA SHOT DETAIL FILE
# ============================================================

def read_shot_detail(file_path):

    if tarfile.is_tarfile(file_path):

        with tarfile.open(file_path, 'r:*') as tar:

            csv_files = [
                member for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith('.csv')
            ]

            if not csv_files:
                raise ValueError("No CSV file found inside the TAR archive.")

            # Prefer a CSV whose name contains shotdetail
            matching_files = [
                member for member in csv_files
                if 'shotdetail' in member.name.lower()
            ]

            if matching_files:
                csv_member = matching_files[0]
            elif len(csv_files) == 1:
                csv_member = csv_files[0]
            else:
                raise ValueError(
                    "Multiple CSV files found. Unable to identify the shot detail file."
                )

            csv_inside = tar.extractfile(csv_member)
            df = pd.read_csv(csv_inside)

    else:
        df = pd.read_csv(file_path)

    return df


# ============================================================
# 2. FUNCTION TO CONVERT SHOT TYPE INTO AN INTEGER
# ============================================================

def convert_shot_type_into_int(a):

    if a == '3PT Field Goal':
        return 1

    elif a == '2PT Field Goal':
        return 0

    else:
        return None


# ============================================================
# 3. FUNCTION TO CATEGORIZE SHOT ACTIONS
# ============================================================

def categorize_action_detailed(action):

    if pd.isna(action):
        return np.nan

    action = str(action)

    if 'Dunk' in action:

        if 'Alley Oop' in action or 'Putback' in action or 'Tip' in action:
            return 'Alley_Oop_Putback_Dunk'

        return 'Standard_Driving_Dunk'

    elif 'Layup' in action or 'Finger Roll' in action:

        if 'Tip' in action or 'Putback' in action:
            return 'Tip_Putback_Layup'

        elif (
            'Driving' in action
            or 'Running' in action
            or 'Cutting' in action
            or 'Alley Oop' in action
        ):
            return 'Driving_Cutting_Layup'

        return 'Standing_Reverse_Layup'

    elif 'Floating' in action:
        return 'Floater'

    elif 'Hook' in action:
        return 'Hook_Shot'

    elif 'Step Back' in action:
        return 'Step_Back_Jumper'

    elif 'Fadeaway' in action or 'Turnaround' in action:
        return 'Fadeaway_Jumper'

    elif 'Pullup' in action or 'Pull-Up' in action or 'Running' in action:
        return 'Pullup_Jumper'

    else:
        return 'Catch_And_Shoot_Jumper'


# ============================================================
# 4. MAIN CLEANING FUNCTION
# ============================================================

def clean_shot_detail(file_path):

    # Read the original dataset
    df = read_shot_detail(file_path)

    print("Original Data Shape:", df.shape)

    # Check that the required columns exist
    required_columns = [
        'MINUTES_REMAINING',
        'SECONDS_REMAINING',
        'LOC_X',
        'LOC_Y',
        'SHOT_TYPE',
        'ACTION_TYPE',
        'SHOT_MADE_FLAG',
        'PLAYER_ID',
        'PLAYER_NAME'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # STEP 1: Drop redundant columns
    # --------------------------------------------------------

    cols_to_drop = [
        'GRID_TYPE',
        'EVENT_TYPE',
        'SHOT_ATTEMPTED_FLAG',
        'TEAM_ID',
        'TEAM_NAME',
        'HTM',
        'VTM',
        'GAME_DATE'
    ]

    df_clean = df.drop(columns=cols_to_drop, errors='ignore')

    # --------------------------------------------------------
    # STEP 2: Convert minutes and seconds into total seconds
    # --------------------------------------------------------

    df_clean = df_clean.assign(
        TIME_REMAINING=(
            df_clean['MINUTES_REMAINING'] * 60
            + df_clean['SECONDS_REMAINING']
        )
    )

    df_clean = df_clean.drop(
        columns=['MINUTES_REMAINING', 'SECONDS_REMAINING']
    )

    # --------------------------------------------------------
    # STEP 3: Calculate accurate shot distance
    # --------------------------------------------------------

    df_clean = df_clean.assign(
        ACCURATE_DISTANCE=(
            (
                df_clean['LOC_X']**2
                + df_clean['LOC_Y']**2
            ) ** (1/2)
        ) / 10
    )

    df_clean = df_clean.drop(
        columns=['SHOT_DISTANCE'],
        errors='ignore'
    )

    # Remove shots beyond 40 feet
    df_clean = df_clean[
        df_clean['ACCURATE_DISTANCE'] <= 40
    ]

    # --------------------------------------------------------
    # STEP 4: Convert shot type into an integer
    # --------------------------------------------------------

    df_clean = df_clean.assign(
        IS_THREE_POINTER=df_clean['SHOT_TYPE'].apply(
            convert_shot_type_into_int
        )
    )

    df_clean = df_clean.drop(columns=['SHOT_TYPE'])

    # --------------------------------------------------------
    # STEP 5: Categorize shot mechanics
    # --------------------------------------------------------

    df_clean = df_clean.assign(
        SHOT_CATEGORY=df_clean['ACTION_TYPE'].apply(
            categorize_action_detailed
        )
    )

    # --------------------------------------------------------
    # STEP 6: Calculate FG% by shot category
    # --------------------------------------------------------

    category_makes = (
        df_clean.groupby('SHOT_CATEGORY')['SHOT_MADE_FLAG']
        .sum()
    )

    category_attempts = (
        df_clean.groupby('SHOT_CATEGORY')['SHOT_MADE_FLAG']
        .count()
    )

    category_fg_table = (
        category_makes / category_attempts
    ).rename('CATEGORY_FG_PCT').reset_index()

    print("\nFG% by Shot Category:")
    print(category_fg_table)

    df_clean = df_clean.merge(
        category_fg_table,
        on='SHOT_CATEGORY'
    )

    # --------------------------------------------------------
    # STEP 7: Calculate FG% by player
    # --------------------------------------------------------

    player_makes = (
        df_clean.groupby(
            ['PLAYER_ID', 'PLAYER_NAME']
        )['SHOT_MADE_FLAG'].sum()
    )

    player_attempts = (
        df_clean.groupby(
            ['PLAYER_ID', 'PLAYER_NAME']
        )['SHOT_MADE_FLAG'].count()
    )

    player_fg_table = (
        player_makes / player_attempts
    ).rename('PLAYER_FG_PCT').reset_index()

    print("\nFG% by Player:")
    print(player_fg_table.head(10))

    df_clean = df_clean.merge(
        player_fg_table,
        on=['PLAYER_ID', 'PLAYER_NAME']
    )

    # --------------------------------------------------------
    # STEP 8: Final cleanup
    # --------------------------------------------------------

    df_clean = df_clean.drop(
        columns=['SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE'],
        errors='ignore'
    )

    df_clean = df_clean.dropna()

    print("\nFinal Cleaned Data Shape:", df_clean.shape)
    print("\nFinal Columns:")
    print(df_clean.columns.tolist())

    # Return the cleaned DataFrame
    return df_clean


# ============================================================
# 5. EXAMPLE USAGE
# ============================================================

df_2025 = clean_shot_detail('shotdetail_2025.tar.xz')
