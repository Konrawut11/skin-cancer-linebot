import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO


# โหลด model (โหลดแค่ครั้งเดียว)
model = YOLO("models/best.pt")

@app.route("/")
def index():
    return "OK"

def predict_yolo(image_bytes):
    """
    รับ bytes ของภาพ, ใช้ YOLOv5 predict class และ confidence
    """
    try:
        img = Image.open(image_bytes)
        results = model(img, size=640)  # ปรับขนาด input ถ้าต้องการ
        # results.print()  # แสดงผลใน console (debug)

        # ดึง class ที่ detect ออกมา
        pred = results.pred[0]  # tensor [n,6] columns: x1,y1,x2,y2,conf,class
        if pred.shape[0] == 0:
            return None, 0.0  # ไม่พบอะไรเลย

        # หา class ที่ confidence สูงสุด
        best_idx = pred[:,4].argmax()
        best_conf = float(pred[best_idx,4])
        best_class = int(pred[best_idx,5])

        return best_class, best_conf

    except Exception as e:
        print(f"YOLO prediction error: {e}")
        return None, 0.0
