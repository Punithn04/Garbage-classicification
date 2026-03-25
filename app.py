import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import gdown

app = Flask(__name__)

# ===== DOWNLOAD MODEL =====
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=1fw8eYt_mWl0IwfgOgSXdFnfxDD1EzzDy"
    gdown.download(url, MODEL_PATH, quiet=False)

print("Model file size:", os.path.getsize(MODEL_PATH))

# ===== LOAD MODEL =====
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

classes = ['cardboard','glass','metal','paper','plastic','trash']

# ===== DATABASE =====
DB_FILE = "db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# ===== PREDICTION FUNCTION =====
def predict_image(img):
    img = img.resize((100, 100))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    index = np.argmax(pred[0])

    return classes[index]

# ===== ROUTES =====

@app.route("/")
def home():
    return "API LIVE"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    img = Image.open(file).convert("RGB")

    result = predict_image(img)

    db = load_db()
    db.append({"type": result})
    save_db(db)

    return jsonify({"prediction": result})

@app.route("/stats")
def stats():
    db = load_db()
    count = {}

    for item in db:
        t = item["type"]
        count[t] = count.get(t, 0) + 1

    return jsonify(count)

@app.route("/forecast")
def forecast():
    return jsonify({
        "alert": "High waste expected tomorrow (trend-based)"
    })

if __name__ == "__main__":
    app.run()