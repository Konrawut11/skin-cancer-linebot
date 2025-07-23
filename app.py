import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from detect_skin import detect_skin_disease
import tempfile
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ตรวจสอบ environment variables
required_env_vars = ["CHANNEL_ACCESS_TOKEN", "CHANNEL_SECRET"]
for var in required_env_vars:
    if not os.environ.get(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# ใช้ ENV ที่ตั้งไว้ใน Railway
line_bot_api = LineBotApi(os.environ["CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["CHANNEL_SECRET"])

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return "Skin Cancer Detection LINE Bot is running!", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    """LINE Bot webhook endpoint"""
    # ดึง X-Line-Signature header
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        logger.error("Missing X-Line-Signature header")
        abort(400)
    
    # ดึง request body
    body = request.get_data(as_text=True)
    logger.info(f"Request body: {body}")
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)
    except LineBotApiError as e:
        logger.error(f"LINE Bot API error: {e}")
        abort(500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        abort(500)
    
    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    """จัดการเมื่อผู้ใช้ส่งรูปภาพมา"""
    try:
        logger.info(f"Received image message from user {event.source.user_id}")
        
        # ดาวน์โหลดรูปภาพ
        message_content = line_bot_api.get_message_content(event.message.id)
        
        # สร้างไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            for chunk in message_content.iter_content():
                tf.write(chunk)
            temp_path = tf.name
        
        logger.info(f"Image saved to temporary file: {temp_path}")
        
        # ตรวจจับโรคผิวหนัง
        try:
            result = detect_skin_disease(temp_path)
            logger.info(f"Detection result: {result}")
        except Exception as e:
            logger.error(f"Error in skin disease detection: {e}")
            result = "เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ กรุณาลองใหม่อีกครั้ง"
        
        # ส่งผลลัพธ์กลับไปยังผู้ใช้
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text=result)
        )
        
        # ลบไฟล์ชั่วคราว
        try:
            os.remove(temp_path)
            logger.info(f"Temporary file removed: {temp_path}")
        except OSError as e:
            logger.warning(f"Failed to remove temporary file: {e}")
            
    except LineBotApiError as e:
        logger.error(f"LINE Bot API error in handle_image: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in handle_image: {e}")

@handler.add(MessageEvent)
def handle_text_message(event):
    """จัดการข้อความทั่วไป"""
    if hasattr(event.message, 'text'):
        user_message = event.message.text.lower()
        
        if any(word in user_message for word in ['สวัสดี', 'hello', 'hi', 'เริ่ม']):
            reply_text = """สวัสดีครับ! 👋

ฉันคือ AI สำหรับตรวจจับโรคผิวหนังเบื้องต้น

📸 วิธีใช้งาน:
ส่งรูปภาพผิวหนังที่ต้องการตรวจสอบมาให้ฉัน
ฉันจะวิเคราะห์และแจ้งผลการตรวจสอบให้คุณทราบ

⚠️ คำเตือน:
- ผลการตรวจนี้เป็นเพียงการประเมินเบื้องต้น
- ไม่สามารถใช้แทนการตรวจวินิจฉัยของแพทย์ได้
- หากมีอาการผิดปกติควรปรึกษาแพทย์ผิวหนัง"""
            
        elif any(word in user_message for word in ['ช่วย', 'help', 'วิธี']):
            reply_text = """📋 วิธีการใช้งาน:

1. ถ่ายรูปหรือส่งรูปภาพผิวหนังที่ต้องการตรวจสอบ
2. รอผลการวิเคราะห์จาก AI (ประมาณ 10-30 วินาที)
3. อ่านผลการตรวจสอบและคำแนะนำ

💡 เทคนิคการถ่ายรูป:
- ถ่ายในที่ที่มีแสงสว่างเพียงพอ
- ถ่ายใกล้ ๆ บริเวณที่ต้องการตรวจ
- หลีกเลี่ยงการสั่นไหวของกล้อง"""
            
        else:
            reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจสอบ 📸

หรือพิมพ์ "ช่วย" เพื่อดูวิธีการใช้งาน"""
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

if __name__ == "__main__":
    # ใช้ PORT ที่ Railway กำหนดให้ หรือ default 5000
    port = int(os.environ.get("PORT", 5000))
    
    logger.info(f"Starting app on port {port}")
    
    # รันแอปพลิเคชัน
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False,  # ปิด debug mode ใน production
        threaded=True  # เปิด threading สำหรับการจัดการ request หลายตัว
    )
