import pandas as pd
from .config import FEATURE_COLUMNS

def preprocess_laps(session):
    """
    Clean and filter lap data from session.
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

    return clean_laps
