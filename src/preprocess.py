import pandas as pd
from .config import FEATURE_COLUMNS
from .lap1predict import lap1predict

def preprocess_laps(session):
    """
    Clean and filter lap data from session, and insert synthetic Lap 1 rows if missing.
    """
    laps = session.laps

    # Drop inaccurate or outlap/inlap laps
    clean_laps = laps[
        (laps["IsAccurate"]) &
        (laps["PitOutTime"].isna()) &
        (laps["PitInTime"].isna())
    ]

    # Merge with weather data using timestamp proximity
    weather = session.weather_data

    clean_laps = pd.merge_asof(
        clean_laps.sort_values("Time"),
        weather.sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("60s")  # allow matching within 1 minute
    )

    # Filter to selected features that actually exist
    available_cols = [col for col in FEATURE_COLUMNS if col in clean_laps.columns]
    clean_laps = clean_laps[available_cols]

    # Insert synthetic Lap 1 rows if missing
    synthetic_rows = lap1predict(clean_laps)
    combined_df = pd.concat([synthetic_rows, clean_laps], ignore_index=True)

    return combined_df
