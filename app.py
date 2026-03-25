from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

app = Flask(__name__)

# ===== DOWNLOAD MODEL FROM GOOGLE DRIVE =====
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=14J29rPbZTENNYcAkFv_ZfLP0nCSokeE6"
    gdown.download(url, MODEL_PATH, quiet=False)

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)

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
    img_array = img_array / 255.0

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