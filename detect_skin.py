import os
import logging

logger = logging.getLogger(__name__)

def detect_skin_disease(image_path):
    """
    ตรวจจับโรคผิวหนังจากรูปภาพ
    พร้อมการจัดการ error หาก ML libraries ไม่พร้อมใช้งาน
    """
    try:
        # ลองโหลด ML libraries
        import torch
        from ultralytics import YOLO
        from PIL import Image
        import numpy as np
        
        # ตรวจสอบว่าไฟล์ model มีอยู่ไหม
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return "❌ ระบบไม่พร้อมใช้งาน: ไม่พบไฟล์ AI model\nกรุณาติดต่อผู้พัฒนาระบบ"
        
        # โหลด YOLO model
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        
        # อ่านและประมวลผลรูปภาพ
        try:
            image = Image.open(image_path)
            results = model(image)
            
            # ประมวลผลลัพธ์
            if len(results) > 0 and len(results[0].boxes) > 0:
                # มีการตรวจพบบางอย่าง
                detections = results[0].boxes
                confidence_scores = detections.conf.tolist()
                class_ids = detections.cls.tolist()
                
                # แปล class IDs เป็นชื่อโรค (ต้องปรับตาม model ของคุณ)
                class_names = {
                    0: "ปกติ",
                    1: "มะเร็งผิวหนัง", 
                    2: "ไฝผิดปกติ",
                    3: "ผิวหนังอักเสบ"
                    # เพิ่มตามจำนวน classes ใน model ของคุณ
                }
                
                result_text = "🔍 ผลการตรวจสอบ:\n\n"
                
                for i, (class_id, confidence) in enumerate(zip(class_ids, confidence_scores)):
                    class_name = class_names.get(int(class_id), "ไม่ทราบ")
                    confidence_percent = confidence * 100
                    
                    if confidence_percent > 50:  # เฉพาะที่มั่นใจมากกว่า 50%
                        result_text += f"• {class_name}: {confidence_percent:.1f}%\n"
                
                result_text += "\n⚠️ คำเตือน:\n"
                result_text += "- ผลนี้เป็นเพียงการประเมินเบื้องต้น\n"
                result_text += "- ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้\n"
                result_text += "- หากมีความกังวลควรปรึกษาแพทย์ผิวหนัง"
                
                return result_text
            
            else:
                return """✅ ไม่พบความผิดปกติที่น่ากังวล

⚠️ หมายเหตุ:
- การตรวจนี้เป็นเพียงการประเมินเบื้องต้น
- หากมีอาการผิดปกติควรปรึกษาแพทย์
- ควรตรวจสุขภาพผิวหนังเป็นประจำ"""
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "❌ เกิดข้อผิดพลาดในการประมวลผลรูปภาพ\nกรุณาลองส่งรูปภาพใหม่"
            
    except ImportError as e:
        logger.error(f"Missing ML dependencies: {e}")
        return """❌ ระบบไม่พร้อมใช้งาน

ขออภัย ระบบ AI ยังไม่พร้อมใช้งาน
กรุณาติดต่อผู้พัฒนาระบบเพื่อตรวจสอบ

ในระหว่างนี้แนะนำให้:
• ปรึกษาแพทย์ผิวหนังโดยตรง
• ถ่ายรูปเก็บไว้เพื่อติดตามอาการ
• หลีกเลี่ยงการสัมผัสแสงแดดมากเกินไป"""
        
    except Exception as e:
        logger.error(f"Unexpected error in detect_skin_disease: {e}")
        return "❌ เกิดข้อผิดพลาดไม่คาดคิด\nกรุณาลองใหม่อีกครั้งหรือติดต่อผู้พัฒนาระบบ"
