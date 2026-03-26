from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
import io
import logging
from functools import wraps
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline as hf_pipeline

# ==================== 1 CONFIGURATION ====================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

swagger = Swagger(app)

limiter = Limiter(
    key_func=lambda: request.headers.get("X-API-Key") or get_remote_address(),
    app=app,
    default_limits=["100 per day"]
)

# ==================== 2 API KEY MANAGEMENT ====================
VALID_API_KEYS = {
    "dev-key-for-your-app-123": "My Beatus Mobile App",
    "dev-key-for-john-web-team-456": "John Web Team",
    "dev-key-for-jane-iot-789": "Jane IoT Project"
}

# ==================== 3 LOAD ML MODELS ====================
print("Loading ML models...")

try:
    sensor_model = joblib.load("mobile_sensors/sensor_model_final.pkl")
    sensor_scaler = joblib.load("mobile_sensors/sensor_scaler.pkl")
    sensor_encoder = joblib.load("mobile_sensors/sensor_label_encoder.pkl")
    print("Sensor model loaded.")
except:
    sensor_model = None
    print("Sensor model not found.")

nlp_classifier = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

COGNITIVE_STATE_LABELS = [
    "focused",
    "relaxed",
    "stressed"
]

print("NLP model loaded.")

# ==================== 4 AUTHENTICATION ====================
def require_api_key(f):

    @wraps(f)
    def decorated(*args, **kwargs):

        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key not in VALID_API_KEYS:

            return jsonify({
                "error": "Unauthorized. Provide valid X-API-Key header."
            }), 401

        logging.info(f"Authorized user: {VALID_API_KEYS[api_key]}")

        return f(*args, **kwargs)

    return decorated


# ==================== 5 HELPER FUNCTIONS ====================

def predict_from_sensor(sensor_input):

    if not sensor_model:
        return None

    try:

        features = [
            'heart_rate',
            'skin_temp',
            'steps',
            'activity_level',
            'ambient_noise',
            'hour_of_day'
        ]

        df = pd.DataFrame([sensor_input], columns=features)

        scaled = sensor_scaler.transform(df)

        probs = sensor_model.predict_proba(scaled)[0]

        return {
            sensor_encoder.classes_[i]: prob
            for i, prob in enumerate(probs)
        }

    except Exception as e:

        logging.error(f"Sensor prediction failed: {e}")
        return None


def predict_from_text(text_input):

    try:

        result = nlp_classifier(
            text_input,
            COGNITIVE_STATE_LABELS,
            multi_label=False
        )

        return {
            label.capitalize(): score
            for label, score in zip(result["labels"], result["scores"])
        }

    except Exception as e:

        logging.error(f"Text prediction failed: {e}")
        return None


def fuse_predictions(predictions, weights):

    final_scores = {
        "Focused": 0.0,
        "Relaxed": 0.0,
        "Stressed": 0.0
    }

    total_weight = 0

    for modality, probs in predictions.items():

        if probs:

            weight = weights.get(modality, 0)

            total_weight += weight

            for state, prob in probs.items():

                final_scores[state] += prob * weight

    if total_weight > 0:

        for state in final_scores:
            final_scores[state] /= total_weight

    final_state = max(final_scores, key=final_scores.get)

    confidence = final_scores[final_state]

    return final_state, confidence


def get_dynamic_freq(state, confidence):

    if state == "Stressed":
        return 10 - 7 * confidence

    if state == "Relaxed":
        return 8 + 2 * confidence

    if state == "Focused":
        return 14 + 8 * confidence

    return 10


def generate_binaural_beat(state, confidence,
                          duration=30,
                          base_freq=200):

    diff = get_dynamic_freq(state, confidence)

    sample_rate = 44100

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    left = np.sin(2 * np.pi * base_freq * t)

    right = np.sin(2 * np.pi * (base_freq + diff) * t)

    stereo = np.stack((left, right), axis=-1)

    buffer = io.BytesIO()

    sf.write(buffer, stereo, sample_rate, format="WAV")

    buffer.seek(0)

    return buffer, diff


# ==================== 6 API ROUTES ====================

@app.route("/")
def home():

    return "Cognitive State Binaural Beat API Running"


@app.route("/recommend", methods=["POST"])
@require_api_key
@limiter.limit("5 per minute")
def recommend():

    """
    Generate Binaural Beat Audio
    ---
    tags:
      - Cognitive API

    consumes:
      - application/json

    parameters:

      - in: header
        name: X-API-Key
        required: true
        schema:
          type: string

      - in: body
        name: body
        required: true
        schema:
          type: object

          properties:

            text_input:
              type: string
              example: "I feel stressed today"

            sensor_input:
              type: object
              properties:

                heart_rate:
                  type: number
                  example: 85

                skin_temp:
                  type: number
                  example: 36.5

                steps:
                  type: number
                  example: 1000

                activity_level:
                  type: number
                  example: 2

                ambient_noise:
                  type: number
                  example: 40

                hour_of_day:
                  type: number
                  example: 14

    responses:

      200:
        description: WAV audio output

      401:
        description: Unauthorized
    """

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Invalid JSON"
            }), 400

        predictions = {}

        # TEXT INPUT
        if data.get("text_input"):

            predictions["text"] = predict_from_text(
                data["text_input"]
            )

        # SENSOR INPUT
        if data.get("sensor_input"):

            predictions["sensor"] = predict_from_sensor(
                data["sensor_input"]
            )

        if not predictions:

            return jsonify({
                "error": "Provide text_input or sensor_input"
            }), 400

        weights = {
            "text": 0.5,
            "sensor": 0.5
        }

        final_state, confidence = fuse_predictions(
            predictions,
            weights
        )

        audio_buffer, diff = generate_binaural_beat(
            final_state,
            confidence
        )

        logging.info(
            f"Prediction: {final_state} ({confidence:.2f})"
        )

        return Response(
            audio_buffer.getvalue(),
            mimetype="audio/wav",
            headers={
                "X-Predicted-State": final_state,
                "X-Confidence": f"{confidence:.4f}",
                "X-Beat-Frequency": f"{diff:.2f}"
            }
        )

    except Exception as e:

        logging.error(e)

        return jsonify({
            "error": str(e)
        }), 500


# ==================== 7 RUN SERVER ====================

if __name__ == "__main__":

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
