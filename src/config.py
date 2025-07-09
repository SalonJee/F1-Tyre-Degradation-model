import os

# === Path Configs ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Add model directory path:
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === FastF1 Cache ===
FASTF1_CACHE_DIR = RAW_DIR

# === Features to keep in processed data ===
FEATURE_COLUMNS = [
    "Driver", "LapTime", "Compound", "TyreLife", "Stint",
    "TrackStatus", "IsAccurate", "Team", "AirTemp", "TrackTemp",
    "LapNumber", "FreshTyre"
]
