from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage, FlexSendMessage
from detect_yolo import predict_yolo
from io import BytesIO
import os

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ตัวอย่าง map class เป็นชื่อโรค (แก้ให้ตรงกับ model ของคุณ)
CLASS_NAMES = {
    0: "Melanoma",
    1: "Nevus",
    2: "Seborrheic Keratosis"
}

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = BytesIO(message_content.content)
    
    predicted_class, confidence = predict_yolo(image_bytes)
    
    if predicted_class is not None:
        label = CLASS_NAMES.get(predicted_class, "Unknown")
        reply = f"ตรวจพบ: {label} (ความมั่นใจ {confidence*100:.2f}%)"
    else:
        reply = "ไม่พบวัตถุที่สามารถวิเคราะห์ได้ กรุณาลองใหม่อีกครั้ง"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))


@app.route('/')
def index():
    return "Skin Cancer Detection LINE Bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
