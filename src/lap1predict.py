# src/lap1predict.py

import pandas as pd
from .config import FEATURE_COLUMNS

def lap1predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every Driver that does not have any Lap 1,
    create a synthetic Lap 1 row using the mean of their available laps.
    """
    synthetic_rows = []

    for driver, group in df.groupby("Driver"):
        if not (group["LapNumber"] == 1).any():
            template = group.iloc[0].copy()

            # Set fixed values for lap 1 with controlled precision
            template["LapNumber"] = 1
            template["TyreLife"] = 1

            template["FreshTyre"] = True
            template["IsAccurate"] = True

            mean_lap_time = group["LapTime"].mean()
            # Round LapTime to milliseconds precision
            template["LapTime"] = mean_lap_time.round('ms')

            # Use mode for TrackStatus or default to 1
            if not group["TrackStatus"].mode().empty:
                template["TrackStatus"] = group["TrackStatus"].mode().iloc[0]
            else:
                template["TrackStatus"] = 1

            template["AirTemp"] = round(group["AirTemp"].mean(), 1)
            template["TrackTemp"] = round(group["TrackTemp"].mean(), 1)

            if not group["Compound"].mode().empty:
                template["Compound"] = group["Compound"].mode().iloc[0]

            template["Stint"] = float(group["Stint"].min())

            synthetic_rows.append(template)

    return pd.DataFrame(synthetic_rows)
