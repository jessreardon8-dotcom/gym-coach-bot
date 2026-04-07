import os
import json
import threading
import time
from collections import defaultdict
import schedule
import anthropic
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_HISTORY = 10
DATA_FILE = "users.json"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Persistent storage ---

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                data = json.load(f)
            known = set(data.get("known_users", []))
            profiles = defaultdict(
                lambda: {"name": None, "goal": None, "onboarded": False, "workouts": []},
                {int(k): v for k, v in data.get("profiles", {}).items()}
            )
            return known, profiles
        except Exception as e:
            print(f"Load error: {e}")
    return set(), defaultdict(lambda: {"name": None, "goal": None, "onboarded": False, "workouts": []})

def save_data():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({
                "known_users": list(known_users),
                "profiles": {str(k): v for k, v in user_profiles.items()}
            }, f)
    except Exception as e:
        print(f"Save error: {e}")

known_users, user_profiles = load_data()
conversation_history = defaultdict(list)

# --- System prompts ---

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
- If they're losing motivation, remind them why they started"""

ONBOARDING_PROMPT = """You are a tough but invested gym coach texting a new client for the first time. Ask their name and their #1 fitness goal in one short punchy message. Be direct and exciting — make them want to answer. One message only, 1-2 sentences."""

# --- Helpers ---

WORKOUT_KEYWORDS = [
    "rep", "set", "kg", "lb", "mile", "km", "squat", "bench", "deadlift",
    "curl", "press", "run", "lift", "workout", "gym", "cardio", "pushup",
    "pullup", "plank", "lunge", "row", "sprint", "hiit", "pr", "personal record",
    "dumbbell", "barbell", "treadmill", "bike", "cycling", "swim", "yoga", "pilates"
]

def extract_workout_data(text):
    lower = text.lower()
    return any(kw in lower for kw in WORKOUT_KEYWORDS)

def build_system_prompt(chat_id):
    profile = user_profiles[chat_id]
    parts = [SYSTEM_PROMPT]
    context = []
    if profile["name"]:
        context.append(f"Client name: {profile['name']}")
    if profile["goal"]:
        context.append(f"Their goal: {profile['goal']}")
    if profile["workouts"]:
        recent = profile["workouts"][-3:]
        context.append(f"Recent workout mentions: {' | '.join(recent)}")
    if context:
        parts.append("\n\nWHAT YOU KNOW ABOUT THIS CLIENT:\n" + "\n".join(context))
    return "\n".join(parts)

def send_telegram(chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10
        )
    except Exception as e:
        print(f"Send error to {chat_id}: {e}")

# --- AI reply ---

def get_ai_reply(chat_id, user_text):
    try:
        profile = user_profiles[chat_id]
        history = conversation_history[chat_id]

        if extract_workout_data(user_text):
            profile["workouts"].append(user_text[:200])
            if len(profile["workouts"]) > 20:
                profile["workouts"] = profile["workouts"][-20:]
            save_data()

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
            save_data()
            return reply

        history.append({"role": "user", "content": user_text})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=build_system_prompt(chat_id),
            messages=history[-MAX_HISTORY:]
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        if len(history) > MAX_HISTORY:
            conversation_history[chat_id] = history[-MAX_HISTORY:]

        if not profile["name"] or not profile["goal"]:
            extract_profile(chat_id, history)

        return reply
    except Exception as e:
        return f"something went wrong: {str(e)}"

def extract_profile(chat_id, history):
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
        for line in response.content[0].text.strip().splitlines():
            if line.startswith("NAME:"):
                val = line.replace("NAME:", "").strip()
                if val.lower() != "null":
                    profile["name"] = val
            elif line.startswith("GOAL:"):
                val = line.replace("GOAL:", "").strip()
                if val.lower() != "null":
                    profile["goal"] = val
        save_data()
    except Exception:
        pass

# --- /stats command ---

def handle_stats(chat_id):
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "Coach")
    goal = profile.get("goal")
    workouts = profile.get("workouts", [])

    lines = [f"📊 YOUR STATS, {name.upper() if name else 'SOLDIER'}"]
    lines.append("")
    if goal:
        lines.append(f"🎯 Goal: {goal}")
    lines.append(f"💪 Workout mentions logged: {len(workouts)}")
    if workouts:
        lines.append("")
        lines.append("Recent activity:")
        for w in workouts[-5:]:
            lines.append(f"• {w[:80]}")
    else:
        lines.append("No workouts logged yet. Get moving!")

    send_telegram(chat_id, "\n".join(lines))

# --- Scheduled messages ---

def send_daily_checkin():
    if not known_users:
        return
    for chat_id in list(known_users):
        profile = user_profiles.get(chat_id, {})
        name = profile.get("name", "")
        goal = profile.get("goal", "your goal")
        greeting = f"{name}! " if name else ""
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            system="You are a savage drill sergeant gym coach sending a morning check-in text. Be aggressive and motivational. 1-2 sentences max. No emojis.",
            messages=[{"role": "user", "content": f"Send a morning check-in to my client whose goal is: {goal}. Address them as {greeting}"}]
        )
        send_telegram(chat_id, message.content[0].text)

def send_gym_reminder():
    days = {0: "Monday", 1: "Tuesday", 3: "Thursday"}
    # Check current UK day (UTC+0 in winter, UTC+1 in summer — Render runs UTC)
    import datetime
    weekday = datetime.datetime.utcnow().weekday()
    day_name = days.get(weekday)
    if not day_name:
        return
    for chat_id in list(known_users):
        profile = user_profiles.get(chat_id, {})
        name = profile.get("name", "")
        name_part = f"{name}. " if name else ""
        text = f"Oi. {name_part}It's {day_name}. Have you been to the gym yet? No excuses."
        send_telegram(chat_id, text)

# Schedule jobs — times are UTC (matches UK/GMT in winter, 1hr off in BST)
schedule.every().day.at("09:00").do(send_daily_checkin)
schedule.every().monday.at("17:00").do(send_gym_reminder)
schedule.every().tuesday.at("17:00").do(send_gym_reminder)
schedule.every().thursday.at("17:00").do(send_gym_reminder)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(30)

threading.Thread(target=run_scheduler, daemon=True).start()

# --- Telegram polling ---

def handle_update(update):
    try:
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        if not chat_id or not text:
            return

        # Register user
        if chat_id not in known_users:
            known_users.add(chat_id)
            save_data()

        if text == "/stats":
            handle_stats(chat_id)
        elif not text.startswith("/"):
            reply = get_ai_reply(chat_id, text)
            send_telegram(chat_id, reply)
    except Exception as e:
        print(f"Error handling update: {e}")

_polling_lock = threading.Lock()
_seen_updates = set()
_SEEN_UPDATES_MAX = 1000

def poll_telegram():
    if not _polling_lock.acquire(blocking=False):
        print("Polling loop already running — refusing to start a second one.")
        return

    print("Polling loop started.")
    offset = None
    try:
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
                    update_id = update["update_id"]
                    if update_id in _seen_updates:
                        offset = update_id + 1
                        continue
                    _seen_updates.add(update_id)
                    # Trim set so it doesn't grow forever
                    if len(_seen_updates) > _SEEN_UPDATES_MAX:
                        oldest = sorted(_seen_updates)[:_SEEN_UPDATES_MAX // 2]
                        for uid in oldest:
                            _seen_updates.discard(uid)
                    handle_update(update)
                    offset = update_id + 1
            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(5)
    finally:
        _polling_lock.release()

# --- Health server ---

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
