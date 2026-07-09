import pytest

from app.chunking import chunk_document, to_chunks


def test_ids_and_pieces():
    chunks = chunk_document("d", "abcdefghij", size=5, overlap=0)
    assert [c["chunk_id"] for c in chunks] == ["d#0", "d#1"]
    assert chunks[0]["text"] == "abcde"
    assert chunks[1]["text"] == "fghij"


def test_overlap_guard_raises():
    with pytest.raises(ValueError):
        chunk_document("d", "abc", size=5, overlap=5)  # size == overlap
    with pytest.raises(ValueError):
        chunk_document("d", "abc", size=5, overlap=10)  # overlap > size


def test_meta_propagation():
    chunks = chunk_document("d", "abcdef", size=3, overlap=0, meta={"page": 2, "source": "f.pdf"})
    assert chunks
    assert all(c["page"] == 2 and c["source"] == "f.pdf" for c in chunks)


def test_overlapping_windows():
    chunks = chunk_document("d", "abcdefgh", size=4, overlap=2)
    assert chunks[0]["text"] == "abcd"
    assert chunks[1]["text"] == "cdef"  # 2글자 겹침


def test_page_strategy_one_chunk_per_page():
    doc = {"doc_id": "vol3", "text": "긴 페이지 전체 텍스트", "page": 12, "source": "vol3.pdf"}
    chunks = to_chunks(doc, strategy="page")
    assert len(chunks) == 1  # 페이지를 더 쪼개지 않음
    assert chunks[0]["chunk_id"] == "vol3#p12"
    assert chunks[0]["page"] == 12 and chunks[0]["source"] == "vol3.pdf"
    assert chunks[0]["text"] == "긴 페이지 전체 텍스트"


def test_char_strategy_still_splits():
    chunks = to_chunks({"doc_id": "d", "text": "abcdefghij"}, strategy="char", size=5, overlap=0)
    assert len(chunks) == 2


def test_pageless_id_is_deterministic_and_content_based():
    # page 없는 레코드: 배치 위치가 아니라 내용 해시 → 호출/순서와 무관하게 같은 id.
    a1 = to_chunks({"doc_id": "d", "text": "내용A"}, strategy="page")
    a2 = to_chunks({"doc_id": "d", "text": "내용A"}, strategy="page")
    b = to_chunks({"doc_id": "d", "text": "내용B"}, strategy="page")
    assert a1[0]["chunk_id"] == a2[0]["chunk_id"]      # 재호출에도 동일(결정론)
    assert a1[0]["chunk_id"] != b[0]["chunk_id"]        # 다른 내용은 다른 id(덮어쓰기 방지)


def test_pageless_long_text_falls_back_to_char():
    # page 전략 + page 없음 + 긴 텍스트 → 통짜 1청크(임베딩 잘림)가 아니라 char 분할.
    chunks = to_chunks({"doc_id": "d", "text": "x" * 1200}, strategy="page", size=500, overlap=0)
    assert len(chunks) > 1


def test_char_strategy_namespaces_by_page_no_collision():
    # 회귀: 같은 doc_id 의 페이지 레코드들을 char 로 쪼개도 chunk_id 가 충돌하지 않는다.
    p1 = to_chunks({"doc_id": "d", "text": "a" * 10, "page": 1}, strategy="char", size=5, overlap=0)
    p2 = to_chunks({"doc_id": "d", "text": "b" * 10, "page": 2}, strategy="char", size=5, overlap=0)
    ids = [c["chunk_id"] for c in p1 + p2]
    assert len(ids) == len(set(ids))                    # 전부 유니크
    assert p1[0]["chunk_id"].startswith("d#p1-")        # 페이지 네임스페이스


def test_empty_text_is_skipped_not_crashing():
    # 빈/공백 레코드는 배치를 죽이지 않고 건너뛴다.
    assert to_chunks({"doc_id": "d", "text": "   "}, strategy="page") == []
    assert to_chunks({"doc_id": "d", "text": ""}, strategy="char") == []
