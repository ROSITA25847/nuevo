from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
import cv2
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# Configuración Telegram
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

# Cargar modelo
model_path = "modelo/impresion.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=False)
model.conf = 0.25

def send_telegram_alert(img, detections):
    filtered = detections[detections["name"].str.lower() != "imprimiendo"]
    if filtered.empty:
        return

    _, buffer = cv2.imencode('.jpg', img)
    files = {"photo": ("detection.jpg", io.BytesIO(buffer))}
    message = "⚠ Error en impresión 3D detectado ⚠\n\n"
    for _, row in filtered.iterrows():
        message += f"🔹 {row['name']} ({row['confidence']:.2f})\n"

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    data = {"chat_id": CHAT_ID, "caption": message}
    requests.post(url, data=data, files=files)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    file = request.files["image"].read()
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = model(img_cv2)
    df = results.pandas().xyxy[0]
    send_telegram_alert(img_cv2, df)

    return jsonify({"detections": df.to_dict(orient="records")})

@app.route("/", methods=["GET"])
def status():
    return jsonify({"status": "Servidor activo", "model_loaded": model is not None})
