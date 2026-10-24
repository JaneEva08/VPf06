"""Диагностика BOT_TOKEN: токен на экран не выводится.

Запуск: python check_token.py
"""
import json
import os
import urllib.error
import urllib.request

from dotenv import dotenv_values

file_token = (dotenv_values(".env").get("BOT_TOKEN") or "")
env_token = os.environ.get("BOT_TOKEN") or ""


def describe(name: str, t: str) -> None:
    print(f"{name}: длина {len(t)}, двоеточие {'есть' if ':' in t else 'НЕТ'}, "
          f"пробелы/кавычки по краям {'ЕСТЬ' if (t != t.strip() or t[:1] in '\'\"' or t[-1:] in '\'\"') else 'нет'}, "
          f"заглушка {'ДА' if 'your-' in t else 'нет'}")


describe("токен в файле .env", file_token)
if env_token:
    describe("переменная Windows BOT_TOKEN", env_token)
    if env_token != file_token:
        print("!! В Windows задана своя BOT_TOKEN, и она отличается от .env — бот возьмёт её, а не токен из .env.")
else:
    print("переменная Windows BOT_TOKEN: не задана")

used = env_token or file_token  # так выберет бот: load_dotenv не перезаписывает переменные окружения
try:
    with urllib.request.urlopen(f"https://api.telegram.org/bot{used}/getMe", timeout=15) as r:
        data = json.load(r)
    print("Telegram принял токен, бот: @" + data["result"]["username"])
except urllib.error.HTTPError as e:
    print(f"Telegram ответил {e.code}: токен, который использует бот, не принят (отозван или неполный).")
except Exception as e:  # noqa: BLE001
    print(f"Не удалось связаться с Telegram: {type(e).__name__}")
