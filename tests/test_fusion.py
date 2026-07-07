from app.retrievers.fusion import reciprocal_rank_fusion


def test_rrf_merges_and_ranks_common_first():
    a = [{"chunk_id": "x", "text": "tx"}, {"chunk_id": "y", "text": "ty"}]
    b = [{"chunk_id": "y", "text": "ty"}, {"chunk_id": "z", "text": "tz"}]
    out = reciprocal_rank_fusion([a, b], k=60, top_n=10)
    ids = [o["chunk_id"] for o in out]
    assert ids[0] == "y"  # 두 리스트 모두에 등장 → 최상위
    assert set(ids) == {"x", "y", "z"}
    assert all("rrf_score" in o for o in out)


def test_rrf_preserves_metadata():
    a = [{"chunk_id": "x", "text": "tx", "page": 5, "source": "f.pdf"}]
    out = reciprocal_rank_fusion([a], k=60, top_n=10)
    assert out[0]["page"] == 5
    assert out[0]["source"] == "f.pdf"


def test_rrf_respects_top_n():
    a = [{"chunk_id": str(i), "text": "t"} for i in range(10)]
    out = reciprocal_rank_fusion([a], k=60, top_n=3)
    assert len(out) == 3
