from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import json

app = Flask(__name__)

DB_FILE = "db.json"

# ===== DATABASE =====
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# ===== LOAD MODEL =====
# TODO: replace this with your model
def predict_image(img):
    return "plastic"

# ===== ROUTES =====

@app.route("/")
def home():
    return "API LIVE"

@app.route("/predict", methods=["POST"])
def predict():
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
        "alert": "High waste expected tomorrow"
    })

if __name__ == "__main__":
    app.run()