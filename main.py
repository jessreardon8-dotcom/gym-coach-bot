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

# Per-user conversation history and workout tracking
conversation_history = defaultdict(list)
user_profiles = defaultdict(lambda: {"name": None, "goal": None, "onboarded": False, "workouts": []})

SYSTEM_PROMPT = """You are an elite personal trainer and drill sergeant rolled into one. You text your clients like a real coach — no fluff, no essays, just fire.

YOUR PERSONALITY:
- Aggressive, motivational, zero tolerance for excuses
- Tough love: you push hard because you believe in them
- Celebrate wins LOUDLY — make them feel like champions
- Call out excuses immediately and redirect to action
- Short punchy replies only. 1-3 sentences MAX. Like real texts.
- Never use bullet points or numbered lists
- Occasional all-caps for emphasis is fine

WHAT YOU TRACK AND USE:
- Their name — use it often
- Their goal — reference it constantly, tie everything back to it
- Their workouts: exercises, weights, reps, sets they mention — bring these up later
- Their patterns: are they consistent? Skipping days? Making progress?
- Their excuses — remember them and call it out if they repeat

COACHING RULES:
- If they're tired: acknowledge it for one second, then push through
- If they skipped: don't let it slide, make them commit to making it up
- If they hit a PR or crushed a workout: go CRAZY, hype them up
- If they mention weight, reps, or exercises: log it mentally and reference it next time
- Never give generic advice — always tie it to what YOU know about THEM
- If they're losing motivation, remind them why they started

ONBOARDING: The system will handle asking for name/goal on first message. Once you have both, coach hard."""

ONBOARDING_PROMPT = """You are a tough but invested gym coach texting a new client for the first time. Ask their name and their #1 fitness goal in one short punchy message. Be direct and exciting — make them want to answer. One message only, 1-2 sentences."""

def extract_workout_data(text):
    """Pull out any workout info worth remembering from a message."""
    keywords = ["rep", "set", "kg", "lb", "mile", "km", "squat", "bench", "deadlift",
                "curl", "press", "run", "lift", "workout", "gym", "cardio", "pushup",
                "pullup", "plank", "lunge", "row", "sprint", "hiit", "pr", "personal record"]
    lower = text.lower()
    return any(kw in lower for kw in keywords)

def get_ai_reply(chat_id, user_text):
    try:
        profile = user_profiles[chat_id]
        history = conversation_history[chat_id]

        # Track workout mentions
        if extract_workout_data(user_text):
            profile["workouts"].append(user_text[:200])
            if len(profile["workouts"]) > 20:
                profile["workouts"] = profile["workouts"][-20:]

        # First message — onboard them
        if not profile["onboarded"] and not history:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system=ONBOARDING_PROMPT,
                messages=[{"role": "user", "content": user_text}]
            )
            reply = response.content[0].text
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            profile["onboarded"] = True
            return reply

        # Build context note about what we know
        context_parts = []
        if profile["name"]:
            context_parts.append(f"Client name: {profile['name']}")
        if profile["goal"]:
            context_parts.append(f"Their goal: {profile['goal']}")
        if profile["workouts"]:
            recent = profile["workouts"][-3:]
            context_parts.append(f"Recent workout mentions: {' | '.join(recent)}")

        system = SYSTEM_PROMPT
        if context_parts:
            system += "\n\nWHAT YOU KNOW ABOUT THIS CLIENT:\n" + "\n".join(context_parts)

        history.append({"role": "user", "content": user_text})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=history[-MAX_HISTORY:]
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        # Trim history
        if len(history) > MAX_HISTORY:
            conversation_history[chat_id] = history[-MAX_HISTORY:]

        # Try to extract name and goal from early conversation
        if not profile["name"] or not profile["goal"]:
            extract_profile(chat_id, history)

        return reply
    except Exception as e:
        return f"something went wrong: {str(e)}"

def extract_profile(chat_id, history):
    """Ask Claude to pull name and goal from the conversation so far."""
    try:
        profile = user_profiles[chat_id]
        if profile["name"] and profile["goal"]:
            return

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        )
        prompt = f"""From this conversation extract:
1. The client's name (or null if not mentioned)
2. Their fitness goal (or null if not mentioned)

Conversation:
{transcript}

Reply in this exact format only:
NAME: <name or null>
GOAL: <goal or null>"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        for line in text.strip().splitlines():
            if line.startswith("NAME:"):
                val = line.replace("NAME:", "").strip()
                if val.lower() != "null":
                    profile["name"] = val
            elif line.startswith("GOAL:"):
                val = line.replace("GOAL:", "").strip()
                if val.lower() != "null":
                    profile["goal"] = val
    except Exception:
        pass

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
