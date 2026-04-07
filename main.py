import os
import re
import json
import threading
import time
import datetime
from collections import defaultdict
import schedule
import anthropic
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_HISTORY = 10
DATA_FILE = "users.json"
CALORIE_GOAL = 1350
WATER_GOAL_ML = 2000

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Persistent storage ---

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                data = json.load(f)
            known = set(data.get("known_users", []))
            profiles = defaultdict(
                lambda: {"name": None, "goal": None, "onboarded": False, "workouts": [], "daily": {}},
                {int(k): v for k, v in data.get("profiles", {}).items()}
            )
            return known, profiles
        except Exception as e:
            print(f"Load error: {e}")
    return set(), defaultdict(lambda: {"name": None, "goal": None, "onboarded": False, "workouts": [], "daily": {}})

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

# --- Daily tracking ---

def get_uk_date():
    """Current date in UTC (≈ UK time; 1hr off during BST)."""
    return datetime.datetime.utcnow().date().isoformat()

def get_daily(chat_id):
    """Return today's daily stats for a user, resetting if it's a new day."""
    profile = user_profiles[chat_id]
    today = get_uk_date()
    if not profile.get("daily") or profile["daily"].get("date") != today:
        profile["daily"] = {
            "date": today,
            "calories": 0,
            "water_ml": 0,
            "pills": False,
            "chlorophyll": False,
        }
        save_data()
    return profile["daily"]

def reset_all_daily_stats():
    today = get_uk_date()
    for chat_id in list(known_users):
        user_profiles[chat_id]["daily"] = {
            "date": today,
            "calories": 0,
            "water_ml": 0,
            "pills": False,
            "chlorophyll": False,
        }
    save_data()
    print("Daily stats reset for all users.")

# --- Tracking detection ---

TRACKING_KEYWORDS = [
    "ate", "eaten", "eat", "eating", "had", "breakfast", "lunch", "dinner",
    "snack", "meal", "coffee", "tea", "drank", "drink", "food", "calorie",
    "burger", "pizza", "sandwich", "salad", "rice", "pasta", "chicken", "fish",
    "fruit", "veg", "protein", "shake", "bar", "biscuit", "cake", "chocolate",
    "water", "ml", "litre", "liter", "glass", "bottle", "hydrat",
    "pill", "tablet", "medication", "supplement", "vitamin", "chlorophyll",
]

def needs_tracking_check(text):
    lower = text.lower()
    return any(kw in lower for kw in TRACKING_KEYWORDS)

def parse_int(s):
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0

def detect_tracking(text):
    """Ask Claude to extract any food/water/pills/chlorophyll from a message."""
    prompt = f"""Analyze this message and extract health tracking info. Reply ONLY in this exact format, no other text:

CALORIES: <estimated calories as integer, 0 if no food mentioned>
WATER_ML: <water in ml as integer, 0 if no water mentioned. 1 small glass=200ml, 1 glass=250ml, 1 large glass=350ml, 1 bottle=500ml, 1 litre=1000ml>
PILLS: <yes or no — only if they explicitly say they took their pills or medication>
CHLOROPHYLL: <yes or no — only if they explicitly say they took chlorophyll>

Rules:
- CALORIES: only estimate if they describe eating specific food. Be accurate. 0 if no food.
- WATER_ML: only if they mention drinking water specifically. 0 for other drinks.
- PILLS/CHLOROPHYLL: must be explicit — "took my pills", "had my chlorophyll", etc.

Message: {text}"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )
        result = {"calories": 0, "water_ml": 0, "pills": False, "chlorophyll": False}
        for line in response.content[0].text.strip().splitlines():
            if line.startswith("CALORIES:"):
                result["calories"] = parse_int(line.split(":", 1)[1])
            elif line.startswith("WATER_ML:"):
                result["water_ml"] = parse_int(line.split(":", 1)[1])
            elif line.startswith("PILLS:"):
                result["pills"] = "yes" in line.lower()
            elif line.startswith("CHLOROPHYLL:"):
                result["chlorophyll"] = "yes" in line.lower()
        return result
    except Exception:
        return {"calories": 0, "water_ml": 0, "pills": False, "chlorophyll": False}

def apply_tracking(chat_id, text):
    """Run tracking detection and update daily stats. Returns a context string for the AI."""
    if not needs_tracking_check(text):
        return None

    tracking = detect_tracking(text)
    daily = get_daily(chat_id)
    updates = []
    changed = False

    if tracking["calories"] > 0:
        daily["calories"] += tracking["calories"]
        remaining = max(0, CALORIE_GOAL - daily["calories"])
        updates.append(f"Logged {tracking['calories']} cal — total {daily['calories']}/{CALORIE_GOAL} ({remaining} remaining)")
        changed = True

    if tracking["water_ml"] > 0:
        daily["water_ml"] += tracking["water_ml"]
        remaining_ml = max(0, WATER_GOAL_ML - daily["water_ml"])
        updates.append(f"Logged {tracking['water_ml']}ml water — total {daily['water_ml']}ml/{WATER_GOAL_ML}ml ({remaining_ml}ml to go)")
        changed = True

    if tracking["pills"] and not daily["pills"]:
        daily["pills"] = True
        updates.append("Pills logged ✅")
        changed = True

    if tracking["chlorophyll"] and not daily["chlorophyll"]:
        daily["chlorophyll"] = True
        updates.append("Chlorophyll logged ✅")
        changed = True

    if changed:
        save_data()

    return " | ".join(updates) if updates else None

# --- Daily summary ---

def format_daily_summary(chat_id, include_savage_comment=False):
    daily = get_daily(chat_id)
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")

    cal_remaining = max(0, CALORIE_GOAL - daily["calories"])
    water_pct = min(100, int(daily["water_ml"] / WATER_GOAL_ML * 100))
    water_remaining = max(0, WATER_GOAL_ML - daily["water_ml"])

    lines = [
        "📊 TODAY'S SUMMARY",
        "",
        f"🔥 Calories: {daily['calories']} / {CALORIE_GOAL} — {cal_remaining} left",
        f"💧 Water: {daily['water_ml']}ml / {WATER_GOAL_ML}ml ({water_pct}%{' — ' + str(water_remaining) + 'ml to go' if water_remaining > 0 else ' — DONE ✅'})",
        f"💊 Pills: {'✅' if daily['pills'] else '❌ not logged'}",
        f"🌿 Chlorophyll: {'✅' if daily['chlorophyll'] else '❌ not logged'}",
    ]

    if include_savage_comment:
        try:
            prompt = f"""You are a savage drill sergeant gym coach. Write ONE punchy sentence (max 20 words) reacting to this person's day. Be specific, brutal if they slacked, hyped if they killed it. Use their name if provided.

Name: {name or 'soldier'}
Calories: {daily['calories']}/{CALORIE_GOAL}
Water: {daily['water_ml']}ml/{WATER_GOAL_ML}ml
Pills taken: {daily['pills']}
Chlorophyll taken: {daily['chlorophyll']}"""
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            lines.append("")
            lines.append(response.content[0].text.strip())
        except Exception:
            pass

    return "\n".join(lines)

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

COACHING RULES:
- Reference their name, goal, and tracked stats when relevant
- If they logged food: acknowledge it and comment on their calories remaining
- If they logged water: acknowledge it and push them to hit 2L
- If pills/chlorophyll not taken by evening: remind them
- If they hit a PR or crushed a workout: go CRAZY, hype them up
- Call out excuses and redirect to action immediately"""

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

def build_system_prompt(chat_id, tracking_update=None):
    profile = user_profiles[chat_id]
    daily = get_daily(chat_id)
    parts = [SYSTEM_PROMPT]
    context = []

    if profile.get("name"):
        context.append(f"Client name: {profile['name']}")
    if profile.get("goal"):
        context.append(f"Their goal: {profile['goal']}")
    if profile.get("workouts"):
        recent = profile["workouts"][-3:]
        context.append(f"Recent workout mentions: {' | '.join(recent)}")

    cal_remaining = max(0, CALORIE_GOAL - daily["calories"])
    water_remaining = max(0, WATER_GOAL_ML - daily["water_ml"])
    context.append(
        f"Today's tracking — Calories: {daily['calories']}/{CALORIE_GOAL} ({cal_remaining} left) | "
        f"Water: {daily['water_ml']}ml/{WATER_GOAL_ML}ml ({water_remaining}ml left) | "
        f"Pills: {'✅' if daily['pills'] else '❌'} | "
        f"Chlorophyll: {'✅' if daily['chlorophyll'] else '❌'}"
    )
    if tracking_update:
        context.append(f"Just logged this message: {tracking_update}")

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

        # Run tracking detection
        tracking_update = apply_tracking(chat_id, user_text)

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
            system=build_system_prompt(chat_id, tracking_update),
            messages=history[-MAX_HISTORY:]
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        if len(history) > MAX_HISTORY:
            conversation_history[chat_id] = history[-MAX_HISTORY:]

        if not profile.get("name") or not profile.get("goal"):
            extract_profile(chat_id, history)

        return reply
    except Exception as e:
        return f"something went wrong: {str(e)}"

def extract_profile(chat_id, history):
    try:
        profile = user_profiles[chat_id]
        if profile.get("name") and profile.get("goal"):
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

# --- Commands ---

def handle_stats(chat_id):
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")
    goal = profile.get("goal")
    workouts = profile.get("workouts", [])

    lines = [f"📊 WORKOUT STATS, {name.upper() if name else 'SOLDIER'}"]
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

def handle_today(chat_id):
    send_telegram(chat_id, format_daily_summary(chat_id, include_savage_comment=True))

# --- Scheduled messages ---

def send_daily_checkin():
    for chat_id in list(known_users):
        profile = user_profiles.get(chat_id, {})
        name = profile.get("name", "")
        goal = profile.get("goal", "your goal")
        greeting = f"{name}! " if name else ""
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system="You are a savage drill sergeant gym coach sending a morning check-in text. Be aggressive and motivational. 1-2 sentences max. No emojis.",
                messages=[{"role": "user", "content": f"Send a morning check-in to my client whose goal is: {goal}. Address them as {greeting}"}]
            )
            send_telegram(chat_id, message.content[0].text)
        except Exception as e:
            print(f"Morning check-in error for {chat_id}: {e}")

def send_gym_reminder():
    days = {0: "Monday", 1: "Tuesday", 3: "Thursday"}
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

def send_evening_checkin():
    for chat_id in list(known_users):
        try:
            summary = format_daily_summary(chat_id, include_savage_comment=True)
            send_telegram(chat_id, summary)
        except Exception as e:
            print(f"Evening check-in error for {chat_id}: {e}")

# Schedule jobs (UTC — matches UK/GMT in winter, 1hr early in BST)
schedule.every().day.at("09:00").do(send_daily_checkin)
schedule.every().day.at("20:00").do(send_evening_checkin)
schedule.every().day.at("00:00").do(reset_all_daily_stats)
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

        if chat_id not in known_users:
            known_users.add(chat_id)
            save_data()

        if text == "/stats":
            handle_stats(chat_id)
        elif text == "/today":
            handle_today(chat_id)
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
