"""팀 계약(1번 파서 입력 / 3번 챗봇 출력) 회귀 테스트."""
import pytest

pytest.importorskip("pydantic_settings")
pytest.importorskip("httpx")
pytest.importorskip("pymilvus")
pytest.importorskip("opensearchpy")

from app.ingestion import _build_chunks  # noqa: E402
from app.main import _to_search_result  # noqa: E402


def test_prechunked_records_pass_through_unchanged():
    # 1번 파서(hyosung_chunks.json) 레코드: chunk_id 가 있으면 재청킹하지 않는다.
    rec = {
        "chunk_id": "hyosung_vol4_p005",
        "doc_id": "hyosung_vol4",
        "source": "효성중공업 전력기술 매거진 Vol.4 (2024)",
        "file_name": "2024_...pdf",
        "page": 5,
        "text": "긴 페이지 본문 " * 100,   # char 청킹이었다면 쪼개졌을 길이
        "extraction": "odl",
    }
    chunks = _build_chunks([rec])
    assert len(chunks) == 1                                  # 재청킹 안 함
    assert chunks[0]["chunk_id"] == "hyosung_vol4_p005"      # 1번 발급 id 보존
    assert chunks[0]["page"] == 5 and chunks[0]["source"].startswith("효성")


def test_prechunked_empty_text_skipped():
    assert _build_chunks([{"chunk_id": "x#1", "doc_id": "x", "text": "  "}]) == []


def test_record_without_chunk_id_is_rejected():
    # pre-chunked 전용 계약: chunk_id 없으면 조용한 오염 대신 즉시 실패.
    with pytest.raises(ValueError, match="chunk_id"):
        _build_chunks([{"doc_id": "raw", "text": "청킹 안 된 원문"}])


def test_search_result_matches_contract_shape():
    # 3번 계약: {chunk_id, text, score, metadata:{doc_id, source, page}}
    hit = {
        "chunk_id": "hyosung_vol4_p005",
        "doc_id": "hyosung_vol4",
        "text": "본문",
        "page": 5,
        "source": "효성중공업 전력기술 매거진 Vol.4 (2024)",
        "rerank_score": 0.8123,
        "rrf_score": 0.03,
    }
    out = _to_search_result(hit)
    assert set(out) == {"chunk_id", "text", "score", "metadata"}
    assert out["score"] == 0.8123                            # 리랭커 점수 우선
    assert set(out["metadata"]) == {"doc_id", "source", "page"}
    assert out["metadata"]["page"] == 5


def test_search_result_score_falls_back_to_rrf():
    out = _to_search_result({"chunk_id": "c", "text": "t", "rrf_score": 0.05})
    assert out["score"] == 0.05                              # 리랭커 off 폴백
