import os
import pandas as pd
import time

from .config import PROCESSED_DIR
from .modeling import load_model, CATEGORICAL, BASE_FEATURES


def list_available_files():
    return [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]


def parse_file_info(filename):
    # Expected format: clean_{gp}({year},{session}).csv
    try:
        main = filename.replace("clean_", "").replace(".csv", "")
        gp, meta = main.split("(")
        year, session = meta.replace(")", "").split(",")
        return int(year), gp, session
    except Exception:
        return None, None, None


def choose_option(prompt, options):
    print(prompt)
    for i, opt in enumerate(options):
        print(f"[{i}] {opt}")
    while True:
        try:
            idx = int(input("Enter choice number: "))
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print("Invalid number.")
        except ValueError:
            print("Enter a valid number.")


def insert_synthetic_first_lap(df, driver, max_lap):
    """
    Inserts synthetic lap 1 with 100% tire health (0 degradation) for the driver,
    extrapolating some features from lap 2 or defaults.
    """
    # Check if lap 1 exists for this driver
    if not ((df["Driver"] == driver) & (df["LapNumber"] == 1)).any():
        # Try to get lap 2 row as template if exists, else first lap for any driver
        template_row = df[(df["Driver"] == driver) & (df["LapNumber"] == 2)]
        if template_row.empty:
            template_row = df[df["LapNumber"] == 1]
            if template_row.empty:
                # Can't build template, skip
                return df

        template_row = template_row.iloc[0].copy()

        # Set lap 1 info
        template_row["LapNumber"] = 1
        template_row["TyreLife"] = 0  # New tire at lap 1
        template_row["PredDegrad"] = 0.0  # No degradation (100% health)
        # LapTime will be predicted later, so set NaN or 0
        template_row["LapTime"] = 0
        template_row["LapTimePred"] = None
        template_row["TotalTime"] = 0

        # Append synthetic lap 1 to dataframe
        df = pd.concat([pd.DataFrame([template_row]), df], ignore_index=True)
        df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    return df


def simulate_race_interactive():
    files = list_available_files()
    if not files:
        print("⚠️ No processed race data available.")
        return

    parsed = [parse_file_info(f) for f in files if parse_file_info(f)[0] is not None]
    if not parsed:
        print("⚠️ No valid files found.")
        return

    years = sorted(set(x[0] for x in parsed))
    year = choose_option("Choose Year:", years)

    gps = sorted(set(x[1] for x in parsed if x[0] == year))
    gp = choose_option("Choose Grand Prix:", gps)

    sessions = sorted(set(x[2] for x in parsed if x[0] == year and x[1] == gp))
    session = choose_option("Choose Session:", sessions)

    file = f"clean_{gp}({year},{session}).csv"
    path = os.path.join(PROCESSED_DIR, file)
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    df = pd.read_csv(path)
    df = df[df["IsAccurate"] == True].copy()
    df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    drivers = sorted(df["Driver"].unique())
    selected_driver = choose_option("Choose Driver:", drivers)

    degr_model = load_model("degradation_model.joblib")
    lap_model = load_model("laptime_model.joblib")

    # Insert synthetic lap 1 with zero degradation for all drivers missing lap 1
    max_lap = df["LapNumber"].max()
    for drv in drivers:
        df = insert_synthetic_first_lap(df, drv, max_lap)

    # Encode base features for all drivers
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    X_encoded = pd.get_dummies(df[BASE_FEATURES], columns=CATEGORICAL, drop_first=True)

    # Align degradation model features
    for col in degr_model.feature_names_in_:
        if col not in X_encoded:
            X_encoded[col] = 0
    X_encoded = X_encoded[degr_model.feature_names_in_]

    # Predict degradation (Note: For synthetic lap 1, PredDegrad=0 from insertion, but can be overwritten)
    df["PredDegrad"] = degr_model.predict(X_encoded)

    # For synthetic lap 1, override PredDegrad to 0 explicitly to represent 100% tire health
    df.loc[df["LapNumber"] == 1, "PredDegrad"] = 0.0

    # Prepare features for lap time model (add PredDegrad)
    X_lap = X_encoded.copy()
    X_lap["PredDegrad"] = df["PredDegrad"]
    for col in lap_model.feature_names_in_:
        if col not in X_lap:
            X_lap[col] = 0
    X_lap = X_lap[lap_model.feature_names_in_]

    df["LapTimePred"] = lap_model.predict(X_lap)

    # Calculate cumulative total time by driver
    df["TotalTime"] = df.groupby("Driver")["LapTimePred"].cumsum()

    print("\n--- Race Simulation ---\n")

    laps_sorted = sorted(df["LapNumber"].unique())

    for lap in laps_sorted:
        lap_df = df[df["LapNumber"] == lap].copy()

        # Selected driver row for this lap
        selected_driver_row = lap_df[lap_df["Driver"] == selected_driver]
        if not selected_driver_row.empty:
            sel = selected_driver_row.iloc[0]
            print(f"Lap {int(lap)}  {selected_driver:<8}  Degrad: {100 - sel['PredDegrad']:.2f}%  LapTime: {sel['LapTimePred']:.3f}s")
        else:
            print(f"Lap {int(lap)}  {selected_driver:<8}  DNF")

        # Print header for all drivers
        print(f"{'Driver':<10} {'Lap':<5} {'LapTime(s)':<12} {'TotalTime(s)':<13} {'Degrad(%)'}")

        # Print all drivers in this lap sorted by TotalTime (fastest first)
        for _, row in lap_df.sort_values("TotalTime").iterrows():
            print(f"{row['Driver']:<10} {int(row['LapNumber']):<5} "
                  f"{row['LapTimePred']:<12.3f} {row['TotalTime']:<13.3f} "
                  f"{100 - row['PredDegrad']:.2f}")
        print()
        time.sleep(0.4)

    print("✅ Simulation complete.")
