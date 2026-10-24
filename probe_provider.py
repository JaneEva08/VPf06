"""Проверка провайдера перед запуском бота: отвечает ли чат-модель и отдаёт ли он эмбеддинги.

Запуск: python probe_provider.py
Нужны OPENAI_API_KEY, OPENAI_BASE_URL (если не OpenAI), CHAT_MODEL, EMBED_MODEL в .env.
"""
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def main() -> None:
    chat_model = os.getenv("CHAT_MODEL", "gpt-5-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    embed = (
        AsyncOpenAI(base_url=os.environ["EMBED_BASE_URL"], api_key=os.getenv("EMBED_API_KEY"))
        if os.getenv("EMBED_BASE_URL")
        else client
    )

    try:
        r = await client.chat.completions.create(
            model=chat_model, messages=[{"role": "user", "content": "Ответь одним словом: привет"}]
        )
        text = r.choices[0].message.content
        print(f"чат {chat_model}: {'OK' if text else 'ПУСТОЙ ОТВЕТ'} -> {text!r}")
        print("usage:", r.usage)
    except Exception as e:  # noqa: BLE001
        print(f"чат {chat_model}: ОШИБКА {type(e).__name__}: {e}")

    try:
        e = await embed.embeddings.create(model=embed_model, input=["Мы делаем сайты", "Офисы в Москве"])
        print(f"эмбеддинги {embed_model}: OK, размерность {len(e.data[0].embedding)}, векторов {len(e.data)}")
    except Exception as e:  # noqa: BLE001
        print(f"эмбеддинги {embed_model}: ОШИБКА {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
