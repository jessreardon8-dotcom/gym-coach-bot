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
DATA_FILE = "data.json"
CALORIE_GOAL = 1350
WATER_GOAL_ML = 2000

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Persistent storage ---

def default_profile():
    return {
        "name": None, "goal": None, "onboarded": False,
        "workouts": [], "daily": {},
        "weight_history": [],   # [{"date": "YYYY-MM-DD", "weight_kg": 65.2}, ...]
        "step_goal": None,
        "weekly_history": [],   # last 7 archived days of daily stats
    }

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                data = json.load(f)
            known = set(data.get("known_users", []))
            profiles = defaultdict(default_profile,
                {int(k): v for k, v in data.get("profiles", {}).items()})
            history = defaultdict(list,
                {int(k): v for k, v in data.get("conversation_history", {}).items()})
            return known, profiles, history
        except Exception as e:
            print(f"Load error: {e}")
    return set(), defaultdict(default_profile), defaultdict(list)

def save_data():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({
                "known_users": list(known_users),
                "profiles": {str(k): v for k, v in user_profiles.items()},
                "conversation_history": {str(k): v for k, v in conversation_history.items()},
            }, f)
    except Exception as e:
        print(f"Save error: {e}")

known_users, user_profiles, conversation_history = load_data()

# --- Daily tracking ---

def get_uk_date():
    return datetime.datetime.utcnow().date().isoformat()

def get_daily(chat_id):
    profile = user_profiles[chat_id]
    today = get_uk_date()
    if not profile.get("daily") or profile["daily"].get("date") != today:
        profile["daily"] = {
            "date": today, "calories": 0, "water_ml": 0,
            "pills": False, "chlorophyll": False, "steps": 0, "gym": False,
        }
        save_data()
    # Backfill steps/gym for profiles saved before these fields existed
    daily = profile["daily"]
    daily.setdefault("steps", 0)
    daily.setdefault("gym", False)
    return daily

def archive_and_reset_daily(chat_id):
    """Save today's stats to weekly_history, then reset for tomorrow."""
    profile = user_profiles[chat_id]
    daily = profile.get("daily", {})
    if daily.get("date"):
        history = profile.setdefault("weekly_history", [])
        history.append(dict(daily))
        profile["weekly_history"] = history[-7:]  # keep last 7 days

    tomorrow = get_uk_date()
    profile["daily"] = {
        "date": tomorrow, "calories": 0, "water_ml": 0,
        "pills": False, "chlorophyll": False, "steps": 0, "gym": False,
    }

def reset_all_daily_stats():
    for chat_id in list(known_users):
        archive_and_reset_daily(chat_id)
    save_data()
    print("Daily stats archived and reset for all users.")

# --- Tracking detection ---

TRACKING_KEYWORDS = [
    "ate", "eaten", "eat", "eating", "had", "breakfast", "lunch", "dinner",
    "snack", "meal", "coffee", "tea", "drank", "drink", "food", "calorie",
    "burger", "pizza", "sandwich", "salad", "rice", "pasta", "chicken", "fish",
    "fruit", "veg", "protein", "shake", "bar", "biscuit", "cake", "chocolate",
    "water", "ml", "litre", "liter", "glass", "bottle", "hydrat",
    "pill", "tablet", "medication", "supplement", "vitamin", "chlorophyll",
    "weigh", "weight", "kg", "stone", "lb", "lbs", "pounds",
    "step", "steps", "walked", "walking",
]

def needs_tracking_check(text):
    lower = text.lower()
    return any(kw in lower for kw in TRACKING_KEYWORDS)

def parse_int(s):
    m = re.search(r'[\d,]+', s)
    return int(m.group().replace(",", "")) if m else 0

def parse_float(s):
    m = re.search(r'[\d.]+', s)
    try:
        return float(m.group()) if m else 0.0
    except ValueError:
        return 0.0

def detect_tracking(text):
    prompt = f"""Analyze this message and extract health tracking info. Reply ONLY in this exact format, no other text:

CALORIES: <estimated calories as integer, 0 if no food mentioned>
WATER_ML: <water in ml as integer, 0 if no water mentioned. 1 small glass=200ml, 1 glass=250ml, 1 large=350ml, 1 bottle=500ml, 1 litre=1000ml>
PILLS: <yes or no — only if they explicitly say they took their pills or medication>
CHLOROPHYLL: <yes or no — only if they explicitly say they took chlorophyll>
WEIGHT_KG: <body weight in kg as decimal, 0 if not mentioned. Convert lbs: divide by 2.205. Convert stones: multiply by 6.35>
STEPS: <step count as integer, 0 if not mentioned. Convert k notation: 10k=10000>

Rules:
- CALORIES: only if they describe eating. Estimate accurately. 0 if none.
- WATER_ML: only for water specifically. 0 for other drinks.
- PILLS/CHLOROPHYLL: must be explicit ("took my pills", "had my chlorophyll").
- WEIGHT_KG: only if they state their body weight. Not food weights.
- STEPS: only if they mention step count or walking distance in steps.

Message: {text}"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        result = {"calories": 0, "water_ml": 0, "pills": False,
                  "chlorophyll": False, "weight_kg": 0.0, "steps": 0}
        for line in response.content[0].text.strip().splitlines():
            if line.startswith("CALORIES:"):
                result["calories"] = parse_int(line.split(":", 1)[1])
            elif line.startswith("WATER_ML:"):
                result["water_ml"] = parse_int(line.split(":", 1)[1])
            elif line.startswith("PILLS:"):
                result["pills"] = "yes" in line.lower()
            elif line.startswith("CHLOROPHYLL:"):
                result["chlorophyll"] = "yes" in line.lower()
            elif line.startswith("WEIGHT_KG:"):
                result["weight_kg"] = parse_float(line.split(":", 1)[1])
            elif line.startswith("STEPS:"):
                result["steps"] = parse_int(line.split(":", 1)[1])
        return result
    except Exception:
        return {"calories": 0, "water_ml": 0, "pills": False,
                "chlorophyll": False, "weight_kg": 0.0, "steps": 0}

def apply_tracking(chat_id, text):
    if not needs_tracking_check(text):
        return None

    tracking = detect_tracking(text)
    daily = get_daily(chat_id)
    profile = user_profiles[chat_id]
    updates = []
    changed = False

    if tracking["calories"] > 0:
        daily["calories"] += tracking["calories"]
        remaining = max(0, CALORIE_GOAL - daily["calories"])
        updates.append(f"Logged {tracking['calories']} cal — total {daily['calories']}/{CALORIE_GOAL} ({remaining} left)")
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

    if tracking["weight_kg"] > 0:
        entry = {"date": get_uk_date(), "weight_kg": round(tracking["weight_kg"], 1)}
        profile.setdefault("weight_history", []).append(entry)
        updates.append(f"Weight logged: {entry['weight_kg']}kg ✅")
        changed = True

    if tracking["steps"] > 0:
        daily["steps"] += tracking["steps"]
        step_goal = profile.get("step_goal")
        if step_goal:
            pct = min(100, int(daily["steps"] / step_goal * 100))
            remaining_steps = max(0, step_goal - daily["steps"])
            if daily["steps"] >= step_goal:
                updates.append(f"Steps: {daily['steps']:,}/{step_goal:,} ✅ GOAL HIT")
            else:
                updates.append(f"Steps: {daily['steps']:,}/{step_goal:,} ({pct}% — {remaining_steps:,} to go)")
        else:
            updates.append(f"Steps logged: {daily['steps']:,}")
        changed = True

    if changed:
        save_data()

    return " | ".join(updates) if updates else None

# --- Daily summary ---

def format_daily_summary(chat_id, include_savage_comment=False):
    daily = get_daily(chat_id)
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")
    step_goal = profile.get("step_goal")
    steps = daily.get("steps", 0)

    cal_remaining = max(0, CALORIE_GOAL - daily["calories"])
    water_pct = min(100, int(daily["water_ml"] / WATER_GOAL_ML * 100))
    water_remaining = max(0, WATER_GOAL_ML - daily["water_ml"])

    lines = [
        "📊 TODAY'S SUMMARY", "",
        f"🔥 Calories: {daily['calories']} / {CALORIE_GOAL} — {cal_remaining} left",
        f"💧 Water: {daily['water_ml']}ml / {WATER_GOAL_ML}ml ({water_pct}%"
        + (f" — {water_remaining}ml to go)" if water_remaining > 0 else " — DONE ✅)"),
        f"💊 Pills: {'✅' if daily['pills'] else '❌ not logged'}",
        f"🌿 Chlorophyll: {'✅' if daily['chlorophyll'] else '❌ not logged'}",
    ]

    if step_goal:
        pct = min(100, int(steps / step_goal * 100))
        lines.append(
            f"👟 Steps: {steps:,} / {step_goal:,} ({pct}%)"
            + (" ✅" if steps >= step_goal else f" — {step_goal - steps:,} to go")
        )
    elif steps > 0:
        lines.append(f"👟 Steps: {steps:,} (no goal set — use /steps [number])")

    if include_savage_comment:
        try:
            prompt = f"""You are a savage drill sergeant gym coach. ONE punchy sentence (max 20 words) on this person's day. Brutal if slacked, hyped if killed it. Use name if provided.

Name: {name or 'soldier'}
Calories: {daily['calories']}/{CALORIE_GOAL}, Water: {daily['water_ml']}ml/{WATER_GOAL_ML}ml
Pills: {daily['pills']}, Chlorophyll: {daily['chlorophyll']}
Steps: {steps}{f'/{step_goal}' if step_goal else ''}"""
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            lines += ["", response.content[0].text.strip()]
        except Exception:
            pass

    return "\n".join(lines)

# --- Weekly summary ---

def format_weekly_summary(chat_id, include_savage_comment=False):
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")
    step_goal = profile.get("step_goal")
    week = profile.get("weekly_history", [])

    if not week:
        return "No weekly data yet — check back after your first full day is tracked."

    days = len(week)
    avg_cal = int(sum(d.get("calories", 0) for d in week) / days)
    avg_water = int(sum(d.get("water_ml", 0) for d in week) / days)
    gym_days = sum(1 for d in week if d.get("gym", False))
    pills_days = sum(1 for d in week if d.get("pills", False))
    chloro_days = sum(1 for d in week if d.get("chlorophyll", False))
    step_days = [d.get("steps", 0) for d in week if d.get("steps", 0) > 0]
    avg_steps = int(sum(step_days) / len(step_days)) if step_days else 0

    lines = [f"📊 WEEKLY SUMMARY ({days} days)", ""]
    lines.append(f"🔥 Avg calories: {avg_cal} / {CALORIE_GOAL}/day")
    lines.append(f"💧 Avg water: {avg_water}ml / {WATER_GOAL_ML}ml/day")
    lines.append(f"💪 Gym sessions: {gym_days} / {days} days")
    lines.append(f"💊 Pills taken: {pills_days} / {days} days")
    lines.append(f"🌿 Chlorophyll: {chloro_days} / {days} days")

    if step_goal and avg_steps > 0:
        lines.append(f"👟 Avg steps: {avg_steps:,} / {step_goal:,} goal")
    elif avg_steps > 0:
        lines.append(f"👟 Avg steps: {avg_steps:,}")

    # Weight trend
    weight_hist = profile.get("weight_history", [])
    week_dates = {d["date"] for d in week}
    week_weights = [w for w in weight_hist if w["date"] in week_dates]
    if len(week_weights) >= 2:
        start_w = week_weights[0]["weight_kg"]
        end_w = week_weights[-1]["weight_kg"]
        diff = round(end_w - start_w, 1)
        arrow = "⬇️" if diff < 0 else ("⬆️" if diff > 0 else "➡️")
        lines.append(f"⚖️ Weight: {start_w}kg → {end_w}kg ({arrow} {abs(diff)}kg)")
    elif len(week_weights) == 1:
        lines.append(f"⚖️ Weight: {week_weights[0]['weight_kg']}kg (log more to see trend)")

    if include_savage_comment:
        try:
            prompt = f"""You are a savage drill sergeant gym coach. Write 2 punchy sentences giving an overall verdict on this person's week. Be specific about what they nailed and what they failed. Use their name.

Name: {name or 'soldier'}
Avg calories: {avg_cal}/{CALORIE_GOAL}, Avg water: {avg_water}ml/{WATER_GOAL_ML}ml
Gym days: {gym_days}/{days}, Pills: {pills_days}/{days}, Chlorophyll: {chloro_days}/{days}
Avg steps: {avg_steps}{f'/{step_goal}' if step_goal else ''}"""
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}]
            )
            lines += ["", response.content[0].text.strip()]
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
- If they logged food: acknowledge and comment on calories remaining
- If they logged water: push them to hit 2L
- If they logged weight: react to it, compare to their goal
- If they logged steps: react to progress vs their step goal
- If pills/chlorophyll not taken: remind them
- If they hit a PR or crushed a workout: go CRAZY, hype them up
- Call out excuses and redirect to action immediately"""

ONBOARDING_PROMPT = """You are a tough but invested gym coach texting a new client for the first time. Ask their name and their #1 fitness goal in one short punchy message. Be direct and exciting — make them want to answer. One message only, 1-2 sentences."""

# --- Helpers ---

WORKOUT_KEYWORDS = [
    "rep", "set", "squat", "bench", "deadlift", "curl", "press", "run",
    "lift", "workout", "gym", "cardio", "pushup", "pullup", "plank", "lunge",
    "row", "sprint", "hiit", "pr", "personal record", "dumbbell", "barbell",
    "treadmill", "cycling", "swim", "yoga", "pilates",
]

def extract_workout_data(text):
    lower = text.lower()
    return any(kw in lower for kw in WORKOUT_KEYWORDS)

def build_system_prompt(chat_id, tracking_update=None):
    profile = user_profiles[chat_id]
    daily = get_daily(chat_id)
    step_goal = profile.get("step_goal")
    steps = daily.get("steps", 0)

    context = []
    if profile.get("name"):
        context.append(f"Client name: {profile['name']}")
    if profile.get("goal"):
        context.append(f"Their goal: {profile['goal']}")
    if profile.get("workouts"):
        context.append(f"Recent workouts: {' | '.join(profile['workouts'][-3:])}")

    weight_hist = profile.get("weight_history", [])
    if weight_hist:
        latest = weight_hist[-1]
        context.append(f"Latest weight: {latest['weight_kg']}kg ({latest['date']})")

    cal_remaining = max(0, CALORIE_GOAL - daily["calories"])
    water_remaining = max(0, WATER_GOAL_ML - daily["water_ml"])
    daily_line = (
        f"Today — Cal: {daily['calories']}/{CALORIE_GOAL} ({cal_remaining} left) | "
        f"Water: {daily['water_ml']}ml/{WATER_GOAL_ML}ml ({water_remaining}ml left) | "
        f"Pills: {'✅' if daily['pills'] else '❌'} | "
        f"Chlorophyll: {'✅' if daily['chlorophyll'] else '❌'} | "
        f"Steps: {steps:,}{f'/{step_goal:,}' if step_goal else ''}"
    )
    context.append(daily_line)
    if tracking_update:
        context.append(f"Just logged: {tracking_update}")

    return SYSTEM_PROMPT + "\n\nWHAT YOU KNOW:\n" + "\n".join(context)

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
            get_daily(chat_id)["gym"] = True

        tracking_update = apply_tracking(chat_id, user_text)

        if not profile.get("onboarded") and not history:
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

        save_data()

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
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": f"""Extract from this conversation:
NAME: <name or null>
GOAL: <goal or null>

{transcript}"""}]
        )
        for line in response.content[0].text.strip().splitlines():
            if line.startswith("NAME:"):
                val = line.split(":", 1)[1].strip()
                if val.lower() != "null":
                    profile["name"] = val
            elif line.startswith("GOAL:"):
                val = line.split(":", 1)[1].strip()
                if val.lower() != "null":
                    profile["goal"] = val
        save_data()
    except Exception:
        pass

# --- Commands ---

def handle_stats(chat_id):
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")
    workouts = profile.get("workouts", [])
    lines = [f"📊 WORKOUT STATS, {name.upper() if name else 'SOLDIER'}", ""]
    if profile.get("goal"):
        lines.append(f"🎯 Goal: {profile['goal']}")
    lines.append(f"💪 Workout mentions logged: {len(workouts)}")
    if workouts:
        lines += ["", "Recent activity:"] + [f"• {w[:80]}" for w in workouts[-5:]]
    else:
        lines.append("No workouts logged yet. Get moving!")
    send_telegram(chat_id, "\n".join(lines))

def handle_today(chat_id):
    send_telegram(chat_id, format_daily_summary(chat_id, include_savage_comment=True))

def handle_week(chat_id):
    send_telegram(chat_id, format_weekly_summary(chat_id, include_savage_comment=True))

def handle_weight(chat_id):
    profile = user_profiles.get(chat_id, {})
    name = profile.get("name", "")
    history = profile.get("weight_history", [])

    if not history:
        send_telegram(chat_id, "No weight logged yet. Tell me your weight and I'll track it.")
        return

    lines = [f"⚖️ WEIGHT HISTORY, {name.upper() if name else 'SOLDIER'}", ""]
    for entry in history[-10:]:
        date = datetime.datetime.strptime(entry["date"], "%Y-%m-%d").strftime("%-d %b")
        lines.append(f"{entry['weight_kg']}kg — {date}")

    if len(history) >= 2:
        diff = round(history[-1]["weight_kg"] - history[0]["weight_kg"], 1)
        if diff < 0:
            trend = f"⬇️ Down {abs(diff)}kg overall"
        elif diff > 0:
            trend = f"⬆️ Up {diff}kg overall"
        else:
            trend = "➡️ Holding steady"
        lines += ["", f"Trend: {trend}"]

        try:
            prompt = f"""You are a savage drill sergeant gym coach. One punchy sentence reacting to this weight trend. Use their name. Be motivational.

Name: {name or 'soldier'}
Start: {history[0]['weight_kg']}kg, Current: {history[-1]['weight_kg']}kg, Change: {diff}kg"""
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            lines += ["", response.content[0].text.strip()]
        except Exception:
            pass

    send_telegram(chat_id, "\n".join(lines))

def handle_steps(chat_id, arg=None):
    profile = user_profiles[chat_id]
    daily = get_daily(chat_id)

    if arg:
        try:
            goal = parse_int(arg)
            if goal > 0:
                profile["step_goal"] = goal
                save_data()
                send_telegram(chat_id, f"Step goal set: {goal:,} steps/day. Now go walk.")
            else:
                send_telegram(chat_id, "That's not a valid step goal. Try /steps 10000")
        except Exception:
            send_telegram(chat_id, "Couldn't parse that. Try /steps 10000")
        return

    step_goal = profile.get("step_goal")
    steps = daily.get("steps", 0)

    if not step_goal:
        send_telegram(chat_id, "No step goal set. Set one with /steps 10000 (or your number).")
        return

    remaining = max(0, step_goal - steps)
    pct = min(100, int(steps / step_goal * 100))
    if steps == 0:
        msg = f"👟 Step goal: {step_goal:,}/day\nToday: nothing logged yet. Move."
    elif steps >= step_goal:
        msg = f"👟 Steps: {steps:,} / {step_goal:,} ✅ GOAL SMASHED"
    else:
        msg = f"👟 Steps: {steps:,} / {step_goal:,} ({pct}%) — {remaining:,} to go"
    send_telegram(chat_id, msg)

# --- Scheduled messages ---

def send_morning_reminder():
    for chat_id in list(known_users):
        profile = user_profiles.get(chat_id, {})
        name = profile.get("name", "")
        greeting = f"{name}! " if name else ""
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system="You are a savage but caring drill sergeant gym coach sending a 7am wake-up text. Remind them to take their pills and chlorophyll. Aggressive and motivational. 1-2 sentences max.",
                messages=[{"role": "user", "content": f"Send a morning reminder to take pills and chlorophyll. Address them as {greeting}"}]
            )
            send_telegram(chat_id, response.content[0].text)
        except Exception as e:
            print(f"Morning reminder error for {chat_id}: {e}")

def send_daily_checkin():
    for chat_id in list(known_users):
        profile = user_profiles.get(chat_id, {})
        name = profile.get("name", "")
        goal = profile.get("goal", "your goal")
        greeting = f"{name}! " if name else ""
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system="You are a savage drill sergeant gym coach sending a morning check-in. Aggressive and motivational. 1-2 sentences max. No emojis.",
                messages=[{"role": "user", "content": f"Send a morning check-in to my client whose goal is: {goal}. Address them as {greeting}"}]
            )
            send_telegram(chat_id, response.content[0].text)
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
        send_telegram(chat_id, f"Oi. {name_part}It's {day_name}. Have you been to the gym yet? No excuses.")

def send_evening_checkin():
    for chat_id in list(known_users):
        try:
            send_telegram(chat_id, format_daily_summary(chat_id, include_savage_comment=True))
        except Exception as e:
            print(f"Evening check-in error for {chat_id}: {e}")

def send_weekly_report():
    for chat_id in list(known_users):
        try:
            send_telegram(chat_id, format_weekly_summary(chat_id, include_savage_comment=True))
        except Exception as e:
            print(f"Weekly report error for {chat_id}: {e}")

# Schedule jobs (UTC — matches UK/GMT in winter, 1hr early in BST)
schedule.every().day.at("07:00").do(send_morning_reminder)
schedule.every().day.at("09:00").do(send_daily_checkin)
schedule.every().day.at("20:00").do(send_evening_checkin)
schedule.every().day.at("00:00").do(reset_all_daily_stats)
schedule.every().monday.at("17:00").do(send_gym_reminder)
schedule.every().tuesday.at("17:00").do(send_gym_reminder)
schedule.every().thursday.at("17:00").do(send_gym_reminder)
schedule.every().sunday.at("18:00").do(send_weekly_report)

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
        elif text == "/week":
            handle_week(chat_id)
        elif text == "/weight":
            handle_weight(chat_id)
        elif text.startswith("/steps"):
            parts = text.split(None, 1)
            arg = parts[1].strip() if len(parts) > 1 else None
            handle_steps(chat_id, arg)
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
