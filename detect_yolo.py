import torch
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


# โหลด model (โหลดแค่ครั้งเดียว)
model = YOLO("models/best.pt")

@app.route("/")
def index():
    return "OK"

def predict_yolo(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))  # ✅ ใช้ BytesIO
        results = model(img, size=640)

        pred = results.pred[0]
        if pred.shape[0] == 0:
            return None, 0.0

        best_idx = pred[:, 4].argmax()
        best_conf = float(pred[best_idx, 4])
        best_class = int(pred[best_idx, 5])

        return best_class, best_conf

    except Exception as e:
        print(f"YOLO prediction error: {e}")
        return None, 0.0
