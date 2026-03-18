import joblib
import numpy as np

import os
import joblib

# Get the directory where this script (model_recommendation.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the model relative to this script
# We go up one level to the project root, then into the models folder
model_path = os.path.join(BASE_DIR, "..", "models", "classification_model.pkl")

model_data = joblib.load(model_path)
model = model_data["model"]
threshold = model_data["threshold"]
features = model_data["features"]

def generate_recommendation(input_dict):
    # Convert input to array using the loaded feature order
    try:
        X = np.array([[input_dict[f] for f in features]])
    except KeyError as e:
        return {"error": f"Missing column in input: {e}"}

    # ML Prediction
    y_prob = model.predict_proba(X)[:, 1]
    prediction = (y_prob >= threshold).astype(int)[0]

    treatments = []

    # WHO-based rule checks
    if input_dict.get("ph", 7) < 6.5 or input_dict.get("ph", 7) > 8.5:
        treatments.append("Boiling / pH Adjustment")

    if input_dict.get("Solids", 0) > 500:
        treatments.append("RO Filtration")

    if input_dict.get("Turbidity", 0) > 5:
        treatments.append("Sediment Filter")

    # If ML says unsafe → suggest UV
    if prediction == 0:
        treatments.append("UV Purification")

    if not treatments:
        treatments.append("No Treatment Required")

    return {
        "Potability_Prediction": "Safe" if prediction == 1 else "Unsafe",
        "Probability": float(y_prob[0]),  # <--- Add [0] here!
        "Recommended_Treatment": " + ".join(sorted(set(treatments)))
    }