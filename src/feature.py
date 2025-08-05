import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from modeling import load_all_data, preprocess_and_engineer, MODEL_DIR

def plot_feature_importance(feature_importance_dict, title="Feature Importance", top_n=15):
    sorted_feats = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_feats)
    y_pos = np.arange(len(features))

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, importances, align='center')
    plt.yticks(y_pos, features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    plt.show()


def get_feature_importance(model_filename, target="degradation"):
    # Load data and preprocess to get feature names
    df = load_all_data()
    df = preprocess_and_engineer(df)

    if target == "degradation":
        X = df.drop(columns=["LapTime", "DegradPct", "BestLapInStint"])
    elif target == "laptime":
        # For lap time model, include PredDegrad — 
        # You'll need to load degradation model to get predictions for PredDegrad, or skip this step
        # For now, assume you just load X without PredDegrad, or handle appropriately
        raise NotImplementedError("Lap time feature importance needs PredDegrad feature preprocessing.")
    else:
        raise ValueError("Invalid target, choose 'degradation' or 'laptime'")

    feature_names = X.columns

    # Load model
    model_path = os.path.join(MODEL_DIR, model_filename)
    model = joblib.load(model_path)

    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))

    print(f"Top feature importances for {target} model:")
    for feat, imp in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat}: {imp:.4f}")

    plot_feature_importance(feature_importance_dict, title=f"{target.capitalize()} Model Feature Importance")


if __name__ == "__main__":
    # Example usage
    get_feature_importance("degradation_model.joblib", target="degradation")
    # For lap time model, you can extend or create a similar function with PredDegrad handling
