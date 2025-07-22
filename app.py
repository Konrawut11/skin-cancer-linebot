from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from detect_skin import detect_skin_disease
import os
import tempfile

app = Flask(__name__)

line_bot_api = LineBotApi("YOUR_CHANNEL_ACCESS_TOKEN")
handler = WebhookHandler("YOUR_CHANNEL_SECRET")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        temp_path = tf.name

    result_text = detect_skin_disease(temp_path)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_text))
    os.remove(temp_path)
