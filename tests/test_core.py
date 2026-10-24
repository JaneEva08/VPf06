import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core  # noqa: E402


def test_chunk_text_overlap_and_limits():
    text = "а" * 1200
    chunks = core.chunk_text(text, 500, 50)
    assert [len(c) for c in chunks] == [500, 500, 300]
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty_and_bad_overlap():
    assert core.chunk_text("   \n ") == []
    with pytest.raises(ValueError):
        core.chunk_text("текст", 50, 50)


def test_company_txt_is_indexed():
    chunks = core.load_document(str(Path(__file__).resolve().parent.parent / "company.txt"))
    assert chunks and any("Москве" in c for c in chunks)


def test_load_document_rejects_unknown_suffix(tmp_path):
    f = tmp_path / "a.exe"
    f.write_text("x")
    with pytest.raises(ValueError):
        core.load_document(str(f))


def test_short_memory_keeps_last_10_and_starts_with_user():
    m = core.ShortMemory()
    for i in range(8):
        m.add_exchange(1, f"q{i}", f"a{i}")
    h = m.history(1)
    assert len(h) == 10
    assert h[0] == {"role": "user", "content": "q3"}
    assert h[-1] == {"role": "assistant", "content": "a7"}


def test_short_memory_is_per_user_and_reset():
    m = core.ShortMemory()
    m.add_exchange(1, "q", "a")
    assert m.history(2) == []
    m.clear(1)
    assert m.history(1) == [] and m.last_user_message(1) is None


@pytest.mark.parametrize("reply", ["", "   \n"])
def test_empty_reply_is_not_stored(reply):
    m = core.ShortMemory()
    with pytest.raises(core.EmptyReplyError):
        m.add_exchange(1, "q", reply)
    assert m.history(1) == []


def test_retrieval_query_adds_previous_question_for_short_followup():
    assert core.build_retrieval_query("А сроки?", "Вы делаете сайты?") == "Вы делаете сайты?\nА сроки?"
    long_q = "Расскажите подробно про безопасность хранения данных клиентов"
    assert core.build_retrieval_query(long_q, "Вы делаете сайты?") == long_q
    assert core.build_retrieval_query("А сроки?", None) == "А сроки?"


def test_filter_by_distance_drops_far_chunks():
    docs = ["близко", "на границе", "далеко"]
    assert core.filter_by_distance(docs, [0.2, 0.75, 0.9], 0.75) == ["близко", "на границе"]
    assert core.filter_by_distance(docs, [0.9, 0.95, 1.0], 0.75) == []


def test_unpack_query_result_handles_empty():
    assert core.unpack_query_result({"documents": [[]], "distances": [[]]}) == ([], [])
    assert core.unpack_query_result({}) == ([], [])
    assert core.unpack_query_result({"documents": [["a"]], "distances": [[0.1]]}) == (["a"], [0.1])


def test_build_messages_with_and_without_context():
    hist = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    plain = core.build_messages(hist, "привет", [])
    assert plain[0]["role"] == "system" and plain[1:3] == hist
    assert plain[-1] == {"role": "user", "content": "привет"}
    rag = core.build_messages(hist, "где офис?", ["Офисы в Москве"])
    assert "<documents>" in rag[-1]["content"] and "Офисы в Москве" in rag[-1]["content"]
    assert rag[-1]["content"].endswith("Вопрос: где офис?")
    assert "не инструкции" in rag[0]["content"]


def test_current_question_is_not_duplicated_in_history():
    m = core.ShortMemory()
    msgs = core.build_messages(m.history(1), "Кто вы?", [])
    m.add_exchange(1, "Кто вы?", "Студия")
    msgs2 = core.build_messages(m.history(1), "Какой был прошлый вопрос?", [])
    assert [x["content"] for x in msgs2 if x["role"] == "user"] == ["Кто вы?", "Какой был прошлый вопрос?"]
    assert len(msgs) == 2


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\win.txt", "win.txt"),
        ("отчёт (1).pdf", "отчёт (1).pdf"),
        ("a<b>|c.txt", "a_b__c.txt"),
        (None, "fallback"),
        ("...", "fallback"),
    ],
)
def test_safe_file_name(raw, expected):
    assert core.safe_file_name(raw, "fallback") == expected


def test_split_for_telegram():
    assert core.split_for_telegram("x" * 5000) == ["x" * 4096, "x" * 904]
    assert core.split_for_telegram("") == ["Пустой ответ от модели."]
