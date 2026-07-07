import math

from app.eval.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d"]
REL = {"c"}


def test_hit_at_k():
    assert hit_at_k(RANKED, REL, 1) == 0.0
    assert hit_at_k(RANKED, REL, 3) == 1.0


def test_mrr_at_k():
    assert abs(mrr_at_k(RANKED, REL, 10) - 1 / 3) < 1e-9  # 첫 정답이 3위


def test_ndcg_at_k():
    assert abs(ndcg_at_k(RANKED, REL, 10) - 1 / math.log2(4)) < 1e-9


def test_recall_and_precision():
    assert recall_at_k(RANKED, {"a", "c"}, 2) == 0.5
    assert precision_at_k(RANKED, {"a", "c"}, 2) == 0.5


def test_empty_relevant_is_zero():
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0
