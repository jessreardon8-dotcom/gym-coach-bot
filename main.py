import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import anthropic
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_ai_reply(user_text):
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        system="You are a savage but supportive gym coach. Reply like a real text message — short, punchy, no fluff. Keep it under 3 sentences.",
        messages=[{"role": "user", "content": user_text}]
    )
    return response.content[0].text

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
