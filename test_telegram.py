#!/usr/bin/env python3
# Quick Telegram connection test
import requests
import config

TOKEN = config.TELEGRAM_TOKEN

# Step 1: verify token is valid
print("Checking token...")
r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=10)
data = r.json()
if data.get("ok"):
    bot = data["result"]
    print(f"Bot found: @{bot['username']} (ID: {bot['id']})")
else:
    print(f"Token invalid: {data}")
    exit()

# Step 2: fetch updates to find your chat ID
print("\nFetching updates to find your chat ID...")
r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=10)
updates = r.json().get("result", [])

if not updates:
    print("\nNo messages found. Please:")
    print("  1. Open Telegram")
    print("  2. Find your bot by username")
    print("  3. Send it any message (e.g. 'hello')")
    print("  4. Run this script again")
else:
    print("\nMessages found! Your chat IDs:")
    for u in updates:
        msg = u.get("message", {})
        chat = msg.get("chat", {})
        print(f"  Chat ID: {chat.get('id')} | Name: {chat.get('first_name')} {chat.get('last_name','')}")
    print(f"\nCopy your Chat ID into config.py as TELEGRAM_CHAT_ID")

# Step 3: try sending a test message to current config chat ID
print(f"\nTrying to send test message to current TELEGRAM_CHAT_ID: {config.TELEGRAM_CHAT_ID} ...")
r = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    json={"chat_id": config.TELEGRAM_CHAT_ID, "text": "Test message from your AI Forex Bot!"},
    timeout=10,
)
result = r.json()
if result.get("ok"):
    print("SUCCESS — check your Telegram!")
else:
    print(f"FAILED — {result.get('description')}")
    print("Your TELEGRAM_CHAT_ID in config.py is wrong.")
