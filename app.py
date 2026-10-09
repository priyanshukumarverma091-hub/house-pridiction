
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "index.html"
MODEL_FILE = BASE_DIR / "house_price_ann.keras"
PREPROCESSOR_FILE = BASE_DIR / "house_price_preprocessor.joblib"

model = None
preprocessor = None

# Must match the California Housing training dataset columns.
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model():
    global model, preprocessor

    if not MODEL_FILE.exists():
        print(f"ERROR: Missing model file: {MODEL_FILE.name}")
        return False

    if not PREPROCESSOR_FILE.exists():
        print(f"ERROR: Missing preprocessor: {PREPROCESSOR_FILE.name}")
        return False

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)

        print("SUCCESS: ANN model loaded.")
        print("SUCCESS: Preprocessor loaded.")
        return True

    except Exception as error:
        model = None
        preprocessor = None
        print(f"ERROR loading model: {error}")
        return False


@app.route("/", methods=["GET"])
def home():
    if not HTML_FILE.exists():
        return "index.html not found in the project folder.", 404

    return send_file(HTML_FILE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "backend": "running",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return jsonify({
            "error": (
                "Model or preprocessor is not loaded. "
                "Check the terminal and required model files."
            )
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send input as a JSON object."}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing input features.",
            "missing_features": missing
        }), 400

    try:
        values = {}

        for feature in FEATURES:
            value = float(data[feature])

            if not np.isfinite(value):
                return jsonify({
                    "error": f"{feature} must be a valid number."
                }), 400

            values[feature] = value

        if values["MedInc"] <= 0:
            return jsonify({"error": "MedInc must be positive."}), 400

        if values["HouseAge"] < 0:
            return jsonify({"error": "HouseAge cannot be negative."}), 400

        if values["AveRooms"] <= 0 or values["AveBedrms"] <= 0:
            return jsonify({"error": "Room values must be positive."}), 400

        if values["Population"] < 0 or values["AveOccup"] <= 0:
            return jsonify({"error": "Population or occupancy is invalid."}), 400

        if not -90 <= values["Latitude"] <= 90:
            return jsonify({"error": "Latitude is invalid."}), 400

        if not -180 <= values["Longitude"] <= 180:
            return jsonify({"error": "Longitude is invalid."}), 400

        # Apply the exact preprocessing fitted during training.
        input_df = pd.DataFrame([values], columns=FEATURES)
        transformed = preprocessor.transform(input_df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed, dtype=np.float32)

        # Model predicts log1p(price); restore original target scale.
        prediction_log = float(model.predict(transformed, verbose=0)[0][0])
        prediction = max(float(np.expm1(prediction_log)), 0.0)

        # California Housing target is in units of $100,000.
        return jsonify({
            "prediction": round(prediction, 4),
            "price_dollars": round(prediction * 100000, 2),
            "currency": "USD",
            "message": "Prediction successful"
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": f"Invalid input: {error}"}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the backend terminal."
        }), 500


if __name__ == "__main__":
    load_model()

    print("Backend: http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/health")

    app.run(host="127.0.0.1", port=5000, debug=True)