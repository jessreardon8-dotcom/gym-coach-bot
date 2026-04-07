import os
import requests
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def get_ai_reply(user_text):
    payload = {"inputs": f"You are a savage but supportive gym coach. Reply like a real text message. User says: {user_text}"}
    response = requests.post(MODEL_URL, headers=headers, json=payload).json()
    try:
        return response[0]["generated_text"]
    except:
        return "hmm... something went wrong. try again."

def handle_message(update: Update, context: CallbackContext):
    user_text = update.message.text
    reply = get_ai_reply(user_text)
    update.message.reply_text(reply)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")
    def log_message(self, format, *args):
        pass

def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

updater = Updater(token=TELEGRAM_TOKEN)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

print("Savage AI Gym Coach is running...")
updater.start_polling()
updater.idle()
