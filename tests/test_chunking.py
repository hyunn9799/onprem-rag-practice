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


def test_page_strategy_without_page_field():
    chunks = to_chunks({"doc_id": "d", "text": "x"}, strategy="page")
    assert chunks[0]["chunk_id"] == "d#0"


def test_page_strategy_pageless_uses_index_for_unique_id():
    # 같은 doc_id 인데 page 가 없어도 index 로 유니크해야 함(#0 덮어쓰기 방지).
    c0 = to_chunks({"doc_id": "d", "text": "a"}, strategy="page", index=0)
    c1 = to_chunks({"doc_id": "d", "text": "b"}, strategy="page", index=1)
    assert c0[0]["chunk_id"] == "d#0"
    assert c1[0]["chunk_id"] == "d#1"
    assert c0[0]["chunk_id"] != c1[0]["chunk_id"]
