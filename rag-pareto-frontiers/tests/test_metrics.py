from src.metrics.metrics import token_f1, retrieval_title_recall_precision

def test_token_f1():
    assert token_f1("reduce latency and cost", "reduce latency and cost") == 1.0
    assert token_f1("latency", "cost") == 0.0

def test_title_recall_precision():
    retrieved = [(1.0, type("C", (), {"title":"A"})()), (0.5, type("C", (), {"title":"B"})())]
    r, p = retrieval_title_recall_precision(retrieved, ["A","C"])
    assert abs(r - 0.5) < 1e-9
    assert abs(p - 0.5) < 1e-9
