import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR, MODEL_DIR


# Shared constants
BASE_FEATURES = [
    "Driver", "Compound", "TyreLife", "Stint", "TrackStatus",
    "Team", "TrackTemp", "AirTemp", "LapNumber", "FreshTyre"
]
CATEGORICAL = ["Driver", "Compound", "Team", "TrackStatus"]
TARGET_DEGRAD = "DegradPct"
TARGET_LAPTIME = "LapTime"


def load_all_data():
    """Load all processed CSVs into a single DataFrame."""
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(f"Processed data directory not found: {PROCESSED_DIR}")
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {PROCESSED_DIR}")

    dfs = [pd.read_csv(os.path.join(PROCESSED_DIR, f)) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)

    req_cols = BASE_FEATURES + [TARGET_LAPTIME, 'IsAccurate']
    missing = [c for c in req_cols if c not in df_all.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")
    return df_all


def preprocess_and_engineer(df):
    df = df[df['IsAccurate']].copy()
    df = df.dropna(subset=BASE_FEATURES + [TARGET_LAPTIME])
    df[TARGET_LAPTIME] = pd.to_timedelta(df[TARGET_LAPTIME]).dt.total_seconds()

    # Base degradation
    df['BestLapInStint'] = df.groupby(['Driver', 'Stint'])[TARGET_LAPTIME].transform('min')
    df[TARGET_DEGRAD] = ((df[TARGET_LAPTIME] - df['BestLapInStint']) / df['BestLapInStint']) * 100

    # Add relative age of the tire in the stint
    df['MaxTyreLifeInStint'] = df.groupby(['Driver', 'Stint'])['TyreLife'].transform('max')
    df['RelativeTyreAge'] = df['TyreLife'] / df['MaxTyreLifeInStint']

    # Optional: softly bias degradation upward at end of stint (model can still override it)
    # df[TARGET_DEGRAD] += (df['RelativeTyreAge'] > 0.9).astype(float) * 5

    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    return df


def train_degradation_model(df):
    """Train model to predict tire degradation percentage."""
    df = preprocess_and_engineer(df)
    X = df.drop(columns=[TARGET_LAPTIME, TARGET_DEGRAD, 'BestLapInStint'])
    y = df[TARGET_DEGRAD]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"✅ Tire Degradation Model — MAE: {mean_absolute_error(y_test, preds):.2f}% | "
          f"R²: {r2_score(y_test, preds):.3f}")
    return model


def train_laptime_model(df, degradation_model):
    """Train model to predict lap time using predicted degradation."""
    df = preprocess_and_engineer(df)
    X_base = df.drop(columns=[TARGET_LAPTIME, TARGET_DEGRAD, 'BestLapInStint'])
    pred_degrad = degradation_model.predict(X_base)
    X = X_base.copy()
    X['PredDegrad'] = pred_degrad
    y = df[TARGET_LAPTIME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"✅ Lap Time Model — MAE: {mean_absolute_error(y_test, preds):.3f}s | "
          f"R²: {r2_score(y_test, preds):.3f}")
    return model


def save_model(model, filename):
    """Save a model (.joblib) to MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"✅ Saved model to {path}")


def load_model(filename):
    """Load a .joblib model from MODEL_DIR."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# Expose constants
__all__ = ["load_all_data", "train_degradation_model", "train_laptime_model",
           "save_model", "load_model", "BASE_FEATURES", "CATEGORICAL"]
