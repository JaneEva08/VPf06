"""Telegram-бот с комбинированной памятью.

Короткая память — последние N сообщений диалога (dict user_id -> deque), живёт в оперативке.
Долгая память — документы (PDF/DOCX/TXT) -> чанки -> эмбеддинги OpenAI -> ChromaDB на диске.
"""
import asyncio
import logging
import os
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, TypedDict

import chromadb
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# -----------------------------
# Константы и настройки
# -----------------------------
PERSIST_DIR = "./memory"  # ChromaDB (долгая память)
UPLOADS_DIR = "./uploads"  # скачанные файлы
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SHORT_MEMORY_SIZE = 10  # последних сообщений (user + assistant) на пользователя
RAG_TOP_K = 5
MAX_FILE_BYTES = 20 * 1024 * 1024  # лимит Telegram на скачивание для ботов
TG_MESSAGE_LIMIT = 4096
ALLOWED_SUFFIXES = {".txt", ".pdf", ".docx"}

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = (
    "Ты — дружелюбный помощник. Если есть контекст из документов — отвечай прежде всего на его основе. "
    "Если контекста недостаточно, используй историю диалога и общие знания, но не выдумывай факты "
    "о документах, которых нет в контексте. Отвечай лаконично."
)


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


# Короткая память: user_id -> кольцевая очередь последних сообщений
short_memory: Dict[int, Deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=SHORT_MEMORY_SIZE))

router = Router()
openai_client: Optional[AsyncOpenAI] = None


# -----------------------------
# Долгая память: документы -> чанки
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Режет текст на пересекающиеся чанки по ~chunk_size символов."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def load_document(file_path: str) -> List[str]:
    """Извлекает текст из PDF/DOCX/TXT и возвращает список чанков."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    elif suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    else:
        raise ValueError("Поддерживаются только PDF, DOCX и TXT.")
    return chunk_text(text)


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name="long_memory", metadata={"hnsw:space": "cosine"})


async def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = await openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


async def embed_chunks(user_id: int, doc_id: str, file_name: str, chunks: List[str]) -> int:
    """Считает эмбеддинги чанков и сохраняет их в Chroma вместе с метаданными."""
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), 100):  # батчами, чтобы не упереться в лимиты API
        vectors.extend(await embed_texts(chunks[i : i + 100]))
    ids = [f"{user_id}:{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {"user_id": str(user_id), "doc_id": doc_id, "file_name": file_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    await asyncio.to_thread(
        get_collection().upsert, ids=ids, embeddings=vectors, metadatas=metadatas, documents=chunks
    )
    return len(chunks)


async def retrieve_context(user_id: int, question: str, top_k: int = RAG_TOP_K) -> List[str]:
    """Ищет в Chroma чанки, ближайшие по смыслу к вопросу (только документы этого пользователя)."""
    q_emb = (await embed_texts([question]))[0]
    result = await asyncio.to_thread(
        get_collection().query,
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"user_id": str(user_id)},
    )
    docs = result.get("documents") or []
    return docs[0] if docs else []


# -----------------------------
# Ответ: RAG + короткая память
# -----------------------------
def build_messages(user_id: int, user_text: str, context_chunks: List[str]) -> List[ChatMessage]:
    if context_chunks:
        context_text = "\n\n".join(f"Фрагмент {i + 1}:\n{c}" for i, c in enumerate(context_chunks))
        content = f"Контекст из документов:\n{context_text}\n\nВопрос: {user_text}"
    else:
        content = user_text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *short_memory[user_id],
        {"role": "user", "content": content},
    ]


async def answer_with_memory(user_id: int, user_text: str) -> str:
    try:
        context_chunks = await retrieve_context(user_id, user_text)
    except Exception:
        logging.exception("RAG retrieve failed")
        context_chunks = []

    # gpt-5-* не поддерживают temperature; лимит токенов — max_completion_tokens
    completion = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=build_messages(user_id, user_text, context_chunks),
        max_completion_tokens=2000,
    )
    reply = completion.choices[0].message.content or ""

    # В историю кладём чистый вопрос (без фрагментов документа), чтобы память не раздувалась
    short_memory[user_id].append({"role": "user", "content": user_text})
    short_memory[user_id].append({"role": "assistant", "content": reply})
    return reply


async def send_long(message: Message, text: str) -> None:
    text = text or "Пустой ответ от модели."
    for i in range(0, len(text), TG_MESSAGE_LIMIT):
        await message.answer(text[i : i + TG_MESSAGE_LIMIT], parse_mode=None)


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
    short_memory.pop(message.from_user.id, None)
    await message.answer("История диалога очищена.")


@router.message(Command("forget"))
async def on_forget(message: Message) -> None:
    await asyncio.to_thread(get_collection().delete, where={"user_id": str(message.from_user.id)})
    await message.answer("Ваши документы удалены из базы.")


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id
    doc = message.document
    file_name = Path(doc.file_name or f"file_{uuid.uuid4().hex}").name  # защита от path traversal
    if Path(file_name).suffix.lower() not in ALLOWED_SUFFIXES:
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
        chunks = await asyncio.to_thread(load_document, str(save_path))
        if not chunks:
            await message.answer("Не удалось извлечь текст из документа.")
            return
        count = await embed_chunks(user_id, doc_id, file_name, chunks)
    except Exception:
        logging.exception("Document processing failed")
        await message.answer("Не удалось обработать документ. Попробуйте ещё раз.")
        return
    await message.answer(f"Документ «{file_name}» сохранён. Чанков: {count}. Теперь задайте вопрос.")


@router.message(F.text & ~F.text.startswith("/"))
async def on_question(message: Message) -> None:
    try:
        reply = await answer_with_memory(message.from_user.id, message.text)
    except Exception:
        logging.exception("Chat completion failed")
        await message.answer("Не удалось получить ответ от модели. Попробуйте позже.")
        return
    await send_long(message, reply)


async def main() -> None:
    global openai_client
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Задайте BOT_TOKEN и OPENAI_API_KEY (см. .env.example).")

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    openai_client = AsyncOpenAI()  # ключ берётся из OPENAI_API_KEY

    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    dp.include_router(router)
    logging.info("Бот запущен. Ожидаю сообщения...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
