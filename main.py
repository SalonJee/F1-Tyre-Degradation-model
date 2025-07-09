import argparse
from src.load_data import load_session
from src.preprocess import preprocess_laps
from src.modeling import (
    load_all_data,
    train_degradation_model,
    train_laptime_model,
    save_model,
    load_model
)
from src.simulation import simulate_race_interactive
from src.config import PROCESSED_DIR
import os


def run_preprocess(year, gp, session_type):
    session = load_session(year, gp, session_type)
    clean_laps = preprocess_laps(session)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filename = f"clean_{gp}({year},{session_type}).csv"
    path = os.path.join(PROCESSED_DIR, filename)
    clean_laps.to_csv(path, index=False)
    print(f"✅ Preprocessed and saved: {path}")


def run_training():
    df = load_all_data()
    print("🔧 Training degradation model...")
    degr_model = train_degradation_model(df)
    save_model(degr_model, "degradation_model.joblib")

    print("🔧 Training lap time model...")
    lap_model = train_laptime_model(df, degr_model)
    save_model(lap_model, "laptime_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Modeling System")
    parser.add_argument("--stage", required=True, choices=["preprocess", "train", "simulate"])
    parser.add_argument("--year", type=int)
    parser.add_argument("--gp", type=str)
    parser.add_argument("--type", type=str, help="Session type: R, Q, FP1")

    args = parser.parse_args()

    if args.stage == "preprocess":
        if args.year is None or args.gp is None or args.type is None:
            raise ValueError("Must provide --year, --gp, and --type for preprocessing.")
        run_preprocess(args.year, args.gp, args.type)

    elif args.stage == "train":
        run_training()

    elif args.stage == "simulate":
        simulate_race_interactive()
