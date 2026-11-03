from fraud_engine.metrics import SCORE_BUCKETS, Metrics


def test_backwards_compatible_observe_without_score():
    m = Metrics()
    m.observe("APPROVE", 1.5)
    text = m.render()
    assert "fre_scored_total 1" in text
    assert "fre_risk_score_count 0" in text
    assert "fre_rule_hits_total{" not in text


def test_rule_hits_counted_once_per_transaction():
    m = Metrics()
    m.observe("BLOCK", 1.0, risk_score=0.9, rule_codes=["R001", "R002", "R001"])
    m.observe("REVIEW", 1.0, risk_score=0.5, rule_codes=["R001"])
    assert m.rule_hits == {"R001": 2, "R002": 1}
    text = m.render()
    assert 'fre_rule_hits_total{rule="R001"} 2' in text
    assert 'fre_rule_hits_total{rule="R002"} 1' in text
    assert "fre_scored_total 2" in text


def test_score_histogram_is_cumulative():
    m = Metrics()
    for s in (0.0, 0.15, 0.5, 0.95, 1.0):
        m.observe("APPROVE", 1.0, risk_score=s)
    text = m.render()
    assert 'fre_risk_score_bucket{le="0.1"} 1' in text
    assert 'fre_risk_score_bucket{le="0.2"} 2' in text
    assert 'fre_risk_score_bucket{le="0.5"} 3' in text
    assert 'fre_risk_score_bucket{le="0.9"} 3' in text
    assert 'fre_risk_score_bucket{le="1.0"} 5' in text
    assert 'fre_risk_score_bucket{le="+Inf"} 5' in text
    assert "fre_risk_score_count 5" in text
    assert "fre_risk_score_sum 2.600000" in text
    counts = [m.score_buckets[b] for b in SCORE_BUCKETS]
    assert counts == sorted(counts)


def test_metrics_export_no_transaction_identifiers():
    m = Metrics()
    m.observe("BLOCK", 1.0, risk_score=0.8, rule_codes=["R002"])
    text = m.render()
    assert "nameOrig" not in text and "nameDest" not in text and "amount" not in text
