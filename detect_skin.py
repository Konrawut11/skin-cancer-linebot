import os
import logging

logger = logging.getLogger(__name__)

# โหลด YOLOv5 model หนึ่งครั้งที่โมดูลโหลด
try:
    import torch
    from PIL import Image
    import numpy as np
    model_path = "models/best.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบโมเดลที่ {model_path}")

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    logger.info("YOLOv5 model loaded successfully")

except Exception as e:
    model = None
    logger.error(f"ไม่สามารถโหลดโมเดล YOLOv5 ได้: {e}")

def detect_skin_disease(image_path):
    """ตรวจจับโรคผิวหนังจากรูปภาพด้วย YOLOv5"""
    if model is None:
        return "❌ ระบบไม่พร้อมใช้งาน: ไม่สามารถโหลดโมเดล AI ได้\nกรุณาติดต่อผู้พัฒนาระบบ"

    try:
        # โหลดภาพ
        image = Image.open(image_path).convert('RGB')

        # รันการตรวจจับ
        results = model(image)

        # ดึงผลลัพธ์เป็น DataFrame
        detections = results.pandas().xyxy[0]
        if detections.empty:
            return """✅ ไม่พบความผิดปกติที่น่ากังวล

⚠️ หมายเหตุ:
- การตรวจนี้เป็นเพียงการประเมินเบื้องต้น
- หากมีอาการผิดปกติควรปรึกษาแพทย์"""

        # แปลชื่อ class
        class_names = {
            0: "ปกติ",
            1: "มะเร็งผิวหนัง", 
            2: "ไฝผิดปกติ",
            3: "ผิวหนังอักเสบ"
            # เพิ่มให้ครบตามโมเดลของคุณ
        }

        result_text = "🔍 ผลการตรวจสอบ:\n\n"
        for _, row in detections.iterrows():
            class_id = int(row['class'])
            confidence = float(row['confidence'])
            if confidence * 100 < 50:
                continue
            class_name = class_names.get(class_id, "ไม่ทราบ")
            result_text += f"• {class_name}: {confidence*100:.1f}%\n"

        result_text += "\n⚠️ คำเตือน:\n- ผลนี้เป็นเพียงการประเมินเบื้องต้น\n- ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้\n- หากมีความกังวลควรปรึกษาแพทย์ผิวหนัง"
        return result_text

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return "❌ เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ\nกรุณาลองใหม่อีกครั้ง"
