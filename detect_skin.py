import os
import logging
from PIL import Image, ImageStat
import numpy as np

logger = logging.getLogger(__name__)

def detect_skin_disease(image_path):
    """
    ตรวจสอบรูปภาพเบื้องต้นโดยไม่ใช้ ML model
    (ชั่วคราวขณะที่รอแก้ปัญหา model size)
    """
    try:
        # เปิดรูปภาพ
        image = Image.open(image_path)
        logger.info(f"Successfully opened image: {image.size}")
        
        # ตรวจสอบคุณภาพภาพเบื้องต้น
        width, height = image.size
        
        # ตรวจสอบขนาดภาพ
        if width < 100 or height < 100:
            return """❌ รูปภาพมีขนาดเล็กเกินไป
            
กรุณาส่งรูปภาพที่มีความละเอียดสูงกว่านี้
เพื่อการตรวจสอบที่แม่นยำ"""

        # แปลงเป็น RGB ถ้าจำเป็น
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # วิเคราะห์สีเบื้องต้น
        stat = ImageStat.Stat(image)
        avg_color = stat.mean
        
        # แปลงเป็น numpy array สำหรับการวิเคราะห์
        img_array = np.array(image)
        
        # วิเคราะห์ความสว่างและคอนทราสต์
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # สร้างรายงานผล
        result_text = """🔍 ผลการตรวจสอบเบื้องต้น:

📊 คุณภาพภาพ:
"""
        
        if brightness < 50:
            result_text += "• ภาพมืดเกินไป - อาจส่งผลต่อการตรวจสอบ\n"
        elif brightness > 200:
            result_text += "• ภาพสว่างเกินไป - อาจมีแสงสะท้อน\n"
        else:
            result_text += "• ความสว่างเหมาะสม ✓\n"
            
        if contrast < 20:
            result_text += "• คอนทราสต์ต่ำ - รายละเอียดอาจไม่ชัดเจน\n"
        else:
            result_text += "• คอนทราสต์เพียงพอ ✓\n"
            
        result_text += f"• ขนาดภาพ: {width}x{height} พิกเซล ✓\n\n"
        
        # คำแนะนำทั่วไป
        result_text += """🤖 สถานะระบบ AI:
ขณะนี้ระบบ AI กำลังอยู่ในระหว่างการปรับปรุง
การตรวจสอบด้วย AI จะพร้อมใช้งานเร็ว ๆ นี้

💡 คำแนะนำเบื้องต้น:
• ติดตามการเปลี่ยนแปลงของผิวหนัง
• หากมีการเปลี่ยนแปลงผิดปกติ ควรปรึกษาแพทย์
• หลีกเลี่ยงแสงแดดจัดในช่วง 10:00-16:00 น.
• ใช้ครีมกันแดด SPF 30 ขึ้นไป

⚠️ คำเตือนสำคัญ:
การตรวจสอบนี้ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้
หากมีความกังวลเกี่ยวกับสุขภาพผิวหนัง
กรุณาปรึกษาแพทย์ผิวหนังโดยตรง"""

        return result_text
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return """❌ เกิดข้อผิดพลาดในการประมวลผลรูปภาพ

กรุณาตรวจสอบว่า:
• ไฟล์เป็นรูปภาพที่ถูกต้อง (JPG, PNG)
• ขนาดไฟล์ไม่เกิน 10 MB
• รูปภาพไม่เสียหาย

แล้วลองส่งใหม่อีกครั้ง"""
