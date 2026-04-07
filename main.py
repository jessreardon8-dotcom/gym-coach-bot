import os
import threading
from collections import defaultdict
import anthropic
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_HISTORY = 10

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-user conversation history: {chat_id: [{"role": ..., "content": ...}, ...]}
conversation_history = defaultdict(list)

SYSTEM_PROMPT = """You are a savage but deeply invested personal gym coach texting your client. You're tough, direct, and don't sugarcoat — but you genuinely care about their progress.

Rules:
- Reply like a real text message: short, punchy, no bullet points, no emojis unless it fits
- Max 2-3 sentences
- Remember everything they've told you in this conversation — their goals, their workouts, their excuses
- If they mention a goal, hold them to it later
- If they tell you what they did at the gym, acknowledge it specifically and push for more
- If they make excuses (tired, busy, sore), call them out but give them a way forward
- Ask follow-up questions to learn their goals, current routine, weak points
- If you don't know their goal yet, ask — you can't coach blind
- Use their name if they've told you it
- Never give generic advice when you have context about them"""

def get_ai_reply(chat_id, user_text):
    try:
        history = conversation_history[chat_id]
        history.append({"role": "user", "content": user_text})

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=history
        )

        reply = message.content[0].text
        history.append({"role": "assistant", "content": reply})

        # Keep only the last MAX_HISTORY messages
        if len(history) > MAX_HISTORY:
            conversation_history[chat_id] = history[-MAX_HISTORY:]

        return reply
    except Exception as e:
        return f"something went wrong: {str(e)}"

def handle_update(update):
    try:
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        if chat_id and text and not text.startswith("/"):
            reply = get_ai_reply(chat_id, text)
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": reply}
            )
    except Exception as e:
        print(f"Error handling update: {e}")

def poll_telegram():
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            response = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params=params,
                timeout=35
            )
            data = response.json()
            for update in data.get("result", []):
                handle_update(update)
                offset = update["update_id"] + 1
        except Exception as e:
            print(f"Polling error: {e}")

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
print("Savage AI Gym Coach is running...")
poll_telegram()
