"""Telegram-бот с комбинированной памятью.

Короткая память — последние N сообщений диалога (в оперативке, см. core.ShortMemory).
Долгая память — документы (PDF/DOCX/TXT) -> чанки -> эмбеддинги -> ChromaDB на диске.
"""
import asyncio
import hashlib
import logging
import os
import shutil
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

import core

load_dotenv()

PERSIST_DIR = "./memory"  # ChromaDB (долгая память)
UPLOADS_DIR = "./uploads"  # временные файлы: после индексации удаляются
MAX_FILE_BYTES = 20 * 1024 * 1024  # лимит Telegram на скачивание для ботов
EMBED_BATCH = 100  # чанков за один запрос эмбеддингов

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
# Эмбеддинги: "api" — через провайдера (EMBED_MODEL), "local" — локальная модель без API и ключей
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "api").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Необязательный лимит ответа. У reasoning-моделей он включает «мысли», поэтому по умолчанию не задан.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "0")) or None

short_memory = core.ShortMemory()
router = Router()
openai_client: Optional[AsyncOpenAI] = None
embed_client: Optional[AsyncOpenAI] = None  # отдельный клиент для эмбеддингов, если провайдер чата их не отдаёт
# Один запрос пользователя за раз: иначе два быстрых сообщения перепутают порядок истории
_user_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def active_embed_model() -> str:
    return LOCAL_EMBED_MODEL if EMBED_PROVIDER == "local" else EMBED_MODEL


def collection_name() -> str:
    """Имя коллекции зависит от модели эмбеддингов: у разных моделей разная размерность векторов,
    и смешивать их в одной коллекции нельзя."""
    digest = hashlib.md5(f"{EMBED_PROVIDER}:{active_embed_model()}".encode()).hexdigest()[:8]
    return f"long_memory_{digest}"


@lru_cache(maxsize=1)
def get_collection():
    """Одно подключение к Chroma на весь процесс."""
    import chromadb

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=collection_name(), metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def get_local_embedder():
    """Локальная модель эмбеддингов (fastembed, ONNX). Первый запуск скачивает модель (~220 МБ)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=LOCAL_EMBED_MODEL)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    if EMBED_PROVIDER == "local":
        embedder = get_local_embedder()
        return await asyncio.to_thread(lambda: [list(map(float, v)) for v in embedder.embed(texts)])
    resp = await (embed_client or openai_client).embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


async def embed_chunks(user_id: int, doc_id: str, file_name: str, chunks: List[str]) -> int:
    """Считает эмбеддинги чанков и сохраняет их в Chroma вместе с метаданными."""
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):  # батчами, чтобы не упереться в лимиты API
        vectors.extend(await embed_texts(chunks[i : i + EMBED_BATCH]))
    ids = [f"{user_id}:{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {"user_id": str(user_id), "doc_id": doc_id, "file_name": file_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    await asyncio.to_thread(
        get_collection().upsert, ids=ids, embeddings=vectors, metadatas=metadatas, documents=chunks
    )
    return len(chunks)


async def retrieve_context(user_id: int, query: str, top_k: int = core.RAG_TOP_K) -> List[str]:
    """Ищет чанки, ближайшие по смыслу к запросу (только документы этого пользователя)."""
    q_emb = (await embed_texts([query]))[0]
    result = await asyncio.to_thread(
        get_collection().query,
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"user_id": str(user_id)},
        include=["documents", "distances"],
    )
    docs, dists = core.unpack_query_result(result)
    logging.info("RAG: найдено %d, расстояния %s", len(docs), [round(d, 3) for d in dists])
    return core.filter_by_distance(docs, dists)


async def answer_with_memory(user_id: int, user_text: str) -> str:
    query = core.build_retrieval_query(user_text, short_memory.last_user_message(user_id))
    try:
        context_chunks = await retrieve_context(user_id, query)
    except Exception:
        logging.exception("RAG retrieve failed")
        context_chunks = []

    kwargs = {}
    if MAX_COMPLETION_TOKENS:
        kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    # gpt-5-* не принимают temperature, поэтому её не передаём
    completion = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=core.build_messages(short_memory.history(user_id), user_text, context_chunks),
        **kwargs,
    )
    reply = completion.choices[0].message.content or ""
    # В историю кладём чистый вопрос (без фрагментов); пустой ответ сюда не попадёт — будет EmptyReplyError
    short_memory.add_exchange(user_id, user_text, reply)
    return reply


def remove_user_uploads(user_id: int) -> None:
    shutil.rmtree(Path(UPLOADS_DIR) / str(user_id), ignore_errors=True)


# -----------------------------
# Хендлеры
# -----------------------------
@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет! Я бот с короткой и долгой памятью.\n"
        "• Пришли PDF/DOCX/TXT — я сохраню его в векторную базу.\n"
        "• Задавай вопросы: отвечу по документам и помню последние 10 реплик.\n"
        "/reset — забыть историю диалога\n"
        "/forget — удалить мои документы из базы"
    )


@router.message(Command("reset"))
async def on_reset(message: Message) -> None:
    short_memory.clear(message.from_user.id)
    await message.answer("История диалога очищена.")


@router.message(Command("forget"))
async def on_forget(message: Message) -> None:
    user_id = message.from_user.id
    await asyncio.to_thread(get_collection().delete, where={"user_id": str(user_id)})
    remove_user_uploads(user_id)
    await message.answer("Ваши документы удалены из базы.")


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id
    doc = message.document
    file_name = core.safe_file_name(doc.file_name, f"file_{uuid.uuid4().hex}")
    if Path(file_name).suffix.lower() not in core.ALLOWED_SUFFIXES:
        await message.answer("Поддерживаются только PDF, DOCX и TXT.")
        return
    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await message.answer("Файл слишком большой (максимум 20 МБ).")
        return

    user_dir = Path(UPLOADS_DIR) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4().hex
    save_path = user_dir / f"{doc_id}_{file_name}"
    try:
        await bot.download(doc, destination=save_path)
        chunks = await asyncio.to_thread(core.load_document, str(save_path))
        if not chunks:
            await message.answer("Не удалось извлечь текст из документа.")
            return
        count = await embed_chunks(user_id, doc_id, file_name, chunks)
    except Exception:
        logging.exception("Document processing failed")
        await message.answer("Не удалось обработать документ. Попробуйте ещё раз.")
        return
    finally:
        save_path.unlink(missing_ok=True)  # в базе остаются только чанки, сам файл не храним
    await message.answer(f"Документ «{file_name}» сохранён. Чанков: {count}. Теперь задайте вопрос.")


@router.message(F.text & ~F.text.startswith("/"))
async def on_question(message: Message) -> None:
    user_id = message.from_user.id
    async with _user_locks[user_id]:
        try:
            reply = await answer_with_memory(user_id, message.text)
        except core.EmptyReplyError:
            logging.warning("Модель вернула пустой ответ")
            await message.answer("Модель вернула пустой ответ. Попробуйте переформулировать вопрос.")
            return
        except Exception:
            logging.exception("Chat completion failed")
            await message.answer("Не удалось получить ответ от модели. Попробуйте позже.")
            return
    for part in core.split_for_telegram(reply):
        await message.answer(part, parse_mode=None)


async def main() -> None:
    global openai_client, embed_client
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # не печатать адреса запросов в консоль
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Задайте BOT_TOKEN и OPENAI_API_KEY (см. .env.example).")

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    # Ключ — из OPENAI_API_KEY; адрес провайдера (aiprimetech и т. п.) — из OPENAI_BASE_URL
    openai_client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    if os.getenv("EMBED_BASE_URL"):  # эмбеддинги у другого провайдера (ключ — EMBED_API_KEY)
        embed_client = AsyncOpenAI(base_url=os.environ["EMBED_BASE_URL"], api_key=os.getenv("EMBED_API_KEY"))

    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    dp.include_router(router)
    logging.info("Бот запущен. Модель: %s, эмбеддинги (%s): %s", CHAT_MODEL, EMBED_PROVIDER, active_embed_model())
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
