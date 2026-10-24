"""Проверка склейки бота на заглушках (OpenAI и Chroma не вызываются). Нужны aiogram и python-dotenv."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("aiogram")
pytest.importorskip("openai")
pytest.importorskip("dotenv")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bot  # noqa: E402
import core  # noqa: E402


class FakeChat:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(bot, "short_memory", core.ShortMemory())


def use_model(monkeypatch, content, chunks):
    chat = FakeChat(content)
    monkeypatch.setattr(bot, "openai_client", SimpleNamespace(chat=SimpleNamespace(completions=chat)))
    seen = []

    async def fake_retrieve(user_id, query, top_k=5):
        seen.append(query)
        return chunks

    monkeypatch.setattr(bot, "retrieve_context", fake_retrieve)
    return chat, seen


def test_answer_uses_context_and_remembers(monkeypatch):
    chat, _ = use_model(monkeypatch, "В Москве и Санкт-Петербурге", ["Офисы в Москве и СПб"])
    reply = asyncio.run(bot.answer_with_memory(1, "Где офисы?"))
    assert reply == "В Москве и Санкт-Петербурге"
    assert "Офисы в Москве" in chat.calls[0]["messages"][-1]["content"]
    assert bot.short_memory.history(1)[0]["content"] == "Где офисы?"  # в истории без фрагментов
    assert "temperature" not in chat.calls[0]


def test_followup_query_includes_previous_question(monkeypatch):
    _, seen = use_model(monkeypatch, "ответ", [])
    asyncio.run(bot.answer_with_memory(1, "Вы делаете сайты?"))
    asyncio.run(bot.answer_with_memory(1, "А сроки?"))
    assert seen == ["Вы делаете сайты?", "Вы делаете сайты?\nА сроки?"]


def test_empty_reply_raises_and_history_stays_clean(monkeypatch):
    use_model(monkeypatch, "", [])
    with pytest.raises(core.EmptyReplyError):
        asyncio.run(bot.answer_with_memory(1, "привет"))
    assert bot.short_memory.history(1) == []


def test_remove_user_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "UPLOADS_DIR", str(tmp_path))
    d = tmp_path / "7"
    d.mkdir()
    (d / "x.txt").write_text("данные")
    bot.remove_user_uploads(7)
    assert not d.exists()
    bot.remove_user_uploads(7)  # повторный вызов не падает


def test_local_embeddings_path(monkeypatch):
    class Vec(list):
        pass

    class FakeEmbedder:
        def embed(self, texts):
            return [Vec([float(len(x)), 1.0]) for x in texts]

    monkeypatch.setattr(bot, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(bot, "get_local_embedder", lambda: FakeEmbedder())
    assert asyncio.run(bot.embed_texts(["аб", "абв"])) == [[2.0, 1.0], [3.0, 1.0]]


def test_collection_name_depends_on_embed_model(monkeypatch):
    monkeypatch.setattr(bot, "EMBED_PROVIDER", "api")
    monkeypatch.setattr(bot, "EMBED_MODEL", "text-embedding-3-small")
    a = bot.collection_name()
    monkeypatch.setattr(bot, "EMBED_PROVIDER", "local")
    b = bot.collection_name()
    assert a != b and a.startswith("long_memory_") and 3 <= len(b) <= 63


def test_threshold_is_read_from_dotenv(tmp_path):
    """Регрессия: core читал RAG_MAX_DISTANCE до load_dotenv(), и значение из .env игнорировалось."""
    import shutil
    import subprocess

    root = Path(__file__).resolve().parent.parent
    for name in ("bot.py", "core.py"):
        shutil.copy(root / name, tmp_path / name)
    (tmp_path / ".env").write_text("RAG_MAX_DISTANCE=0.5\n", encoding="utf-8")
    env = {k: v for k, v in __import__("os").environ.items() if k != "RAG_MAX_DISTANCE"}
    env.update(BOT_TOKEN="x", OPENAI_API_KEY="x")
    out = subprocess.run(
        [sys.executable, "-c", "import bot, core; print(core.RAG_MAX_DISTANCE)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "0.5"


def test_reupload_replaces_previous_version(monkeypatch):
    calls = []

    class FakeCollection:
        def delete(self, where):
            calls.append(("delete", where))

        def upsert(self, **kwargs):
            calls.append(("upsert", len(kwargs["documents"])))

    async def fake_embed(texts):
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(bot, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(bot, "embed_texts", fake_embed)
    count = asyncio.run(bot.embed_chunks(5, "doc1", "company.txt", ["а", "б"]))
    assert count == 2
    assert calls[0] == ("delete", {"$and": [{"user_id": "5"}, {"file_name": "company.txt"}]})
    assert calls[1] == ("upsert", 2)
