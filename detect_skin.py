from ultralytics import YOLO
from PIL import Image
import torch

# โหลดโมเดล
model = YOLO("models/best.pt")

# ฟังก์ชันรับรูป (จาก path หรือ Image object) แล้วส่งผลลัพธ์เป็น label
def detect_skin_disease(image_path):
    results = model(image_path)
    names = model.names  # dict: {0: "Acne", 1: "Warts", ...}
    
    detected_classes = set()
    for result in results:
        for cls_id in result.boxes.cls.tolist():
            detected_classes.add(names[int(cls_id)])
    
    if detected_classes:
        return f"ตรวจพบ: {', '.join(detected_classes)}"
    else:
        return "ไม่ตรวจพบโรคผิวหนัง"
