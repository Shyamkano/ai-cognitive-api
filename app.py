from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
import os
import io
import base64
import logging
from functools import wraps
import cv2
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import pipeline as hf_pipeline
from deepface import DeepFace

# ==================== 1️⃣ CONFIGURATION ====================
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

swagger = Swagger(app)

limiter = Limiter(
    key_func=lambda: request.headers.get("X-API-Key") or get_remote_address(),
    app=app,
    default_limits=["100 per day"]
)

# ==================== 2️⃣ API KEY MANAGEMENT ====================
# In a real production app, this would be a secure database lookup.
# For your project, this is a perfectly acceptable and effective method.
VALID_API_KEYS = {
    "dev-key-for-your-app-123": "My Beatus Mobile App",
    "dev-key-for-john-web-team-456": "John's Web Team",
    "dev-key-for-jane-iot-789": "Jane's IoT Project"
}

# ==================== 3️⃣ LOAD ML MODELS ====================
print("Loading all machine learning models...")
try:
    sensor_model = joblib.load("mobile_sensors/sensor_model_final.pkl")
    sensor_scaler = joblib.load("mobile_sensors/sensor_scaler.pkl")
    sensor_encoder = joblib.load("mobile_sensors/sensor_label_encoder.pkl")
    print("✅ Sensor Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load Sensor Model: {e}")
    sensor_model = None

nlp_classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
COGNITIVE_STATE_LABELS = ["focused", "relaxed", "stressed"]
print("✅ NLP Model (Zero-Shot) loaded successfully.")

print("Building Facial Emotion Recognition model...")
try:
    DeepFace.build_model("VGG-Face")
    print("✅ Facial Emotion Recognition Model built successfully.")
except Exception as e:
    print(f"⚠️ Could not build Facial Emotion Recognition model: {e}")

EMOTION_TO_STATE_MAP = {
    'happy': 'Focused', 'neutral': 'Relaxed', 'sad': 'Stressed',
    'angry': 'Stressed', 'fear': 'Stressed', 'disgust': 'Stressed', 'surprise': 'Focused'
}

# ==================== 4️⃣ AUTHENTICATION DECORATOR ====================
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in VALID_API_KEYS:
            logging.warning(f"Unauthorized API call attempt with key: {api_key}")
            return jsonify({"error": "Unauthorized. A valid 'X-API-Key' header is required."}), 401
        logging.info(f"Request authorized for: {VALID_API_KEYS[api_key]}")
        return f(*args, **kwargs)
    return decorated

# ==================== 5️⃣ HELPER FUNCTIONS ====================
# ... (All your helper functions: predict_from_sensor, predict_from_text, 
#      predict_from_face, fuse_predictions, get_dynamic_freq, 
#      generate_binaural_beat remain EXACTLY THE SAME) ...

# (For completeness, I am including them here again)
def predict_from_sensor(sensor_input):
    if not sensor_model: return None
    try:
        features = ['heart_rate', 'skin_temp', 'steps', 'activity_level', 'ambient_noise', 'hour_of_day']
        df = pd.DataFrame([sensor_input], columns=features)
        scaled_df = sensor_scaler.transform(df)
        probabilities = sensor_model.predict_proba(scaled_df)[0]
        return {sensor_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}
    except Exception as e:
        logging.error(f"Sensor prediction failed: {e}")
        return None

def predict_from_text(text_input):
    try:
        result = nlp_classifier(text_input, COGNITIVE_STATE_LABELS, multi_label=False)
        return {label.capitalize(): score for label, score in zip(result['labels'], result['scores'])}
    except Exception as e:
        logging.error(f"Text prediction failed: {e}")
        return None

def predict_from_face(base64_image_string):
    try:
        img_data = base64.b64decode(base64_image_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        predicted_state = EMOTION_TO_STATE_MAP.get(dominant_emotion, "Relaxed")
        return {label: (1.0 if label == predicted_state else 0.0) for label in ["Focused", "Relaxed", "Stressed"]}
    except Exception as e:
        logging.error(f"Face prediction failed: {e}")
        return None

def fuse_predictions(predictions, weights):
    final_scores = {"Focused": 0.0, "Relaxed": 0.0, "Stressed": 0.0}
    total_weight = 0.0
    for modality, prob_dict in predictions.items():
        if prob_dict:
            weight = weights.get(modality, 0)
            total_weight += weight
            for state, prob in prob_dict.items():
                final_scores[state] += prob * weight
    if total_weight > 0:
        for state in final_scores:
            final_scores[state] /= total_weight
    final_prediction = max(final_scores, key=final_scores.get)
    confidence = final_scores[final_prediction]
    return final_prediction, confidence

def get_dynamic_freq(state, confidence):
    if state == "Stressed": return 10 - 7 * confidence
    elif state == "Relaxed": return 8 + 2 * confidence
    elif state == "Focused": return 14 + 8 * confidence
    else: return 10

def generate_binaural_beat(state, confidence, duration=30, base_freq=200):
    diff = get_dynamic_freq(state, confidence)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + diff) * t)
    stereo = np.stack((left, right), axis=-1)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer, diff

# ==================== 6️⃣ API ROUTES ====================

@app.route('/')
def home():
    return "🚀 Public Multi-Modal Cognitive State API is running. See documentation for usage."

@app.route('/recommend', methods=['POST'])
@require_api_key
@limiter.limit("5 per minute")
def recommend():
    """
    Generate Adaptive Binaural Beat Stimulus
    ---
    tags:
      - Cognitive AI API
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
            face_image_base64:
              type: string
              example: "BASE64_IMAGE_STRING"
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
        description: WAV audio stream with predicted cognitive state
      400:
        description: Invalid input
      401:
        description: Unauthorized
      429:
        description: Rate limit exceeded
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        predictions = {}

        # 🔹 TEXT INPUT
        if data.get("text_input"):
            predictions['text'] = predict_from_text(data["text_input"])

        # 🔹 IMAGE BASE64 INPUT
        if data.get("face_image_base64"):
            base64_img = data["face_image_base64"]

            # Remove header if frontend sends:
            # data:image/jpeg;base64,...
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            predictions['face'] = predict_from_face(base64_img)

        # 🔹 SENSOR INPUT
        if data.get("sensor_input"):
            predictions['sensor'] = predict_from_sensor(data["sensor_input"])

        if not predictions:
            return jsonify({
                "error": "At least one input (text_input, face_image_base64, or sensor_input) is required."
            }), 400

        weights = {'text': 0.25, 'face': 0.4, 'sensor': 0.35}
        final_state, confidence = fuse_predictions(predictions, weights)

        audio_buffer, diff = generate_binaural_beat(final_state, confidence)

        logging.info(
            f"Final Prediction: {final_state} "
            f"with confidence {confidence:.2f}"
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
        logging.error(f"Error in /recommend: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== 7️⃣ RUN APP ====================
if __name__ == '__main__':
    # For production, use a proper web server like Gunicorn or Waitress
    app.run(host="0.0.0.0", port=5000, debug=False)