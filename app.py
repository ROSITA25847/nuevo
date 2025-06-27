from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
import os

# Telegram config
TOKEN = "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo"
CHAT_ID = "-1002221266716"

app = Flask(__name__)

# Cargar modelo solo una vez
MODEL_PATH = "modelo/impresion.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=False)
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

def send_telegram_alert(image, detections):
    filtered = detections[detections['name'].str.lower() != 'imprimiendo']
    if filtered.empty:
        return

    _, buffer = cv2.imencode(".jpg", image)
    photo_bytes = BytesIO(buffer)
    photo_bytes.seek(0)
    files = {'photo': ('detection.jpg', photo_bytes)}
    
    message = "⚠ Detección de error en impresión 3D ⚠\n\n"
    for _, row in filtered.iterrows():
        message += f"🔹 {row['name']}\nConfianza: {row['confidence']:.2f}\nPosición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    data = {"chat_id": CHAT_ID, "caption": message}
    requests.post(url, data=data, files=files)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        result = model(img_bgr)
        detections = result.pandas().xyxy[0]

        if len(detections) > 0:
            send_telegram_alert(np.squeeze(result.render()), detections)

        return jsonify({"detections": detections.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Servidor de detección 3D en línea."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
