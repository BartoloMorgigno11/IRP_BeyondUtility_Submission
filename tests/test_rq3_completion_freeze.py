from pathlib import Path
import pytest
from src.analysis import rq3_completion_scientific_freeze as freeze
from src.analysis.validate_rq3_completion_freeze import require


def test_full_reciprocal_rank_and_top_five_utility():
    record = {"strict_parse_valid": True, "terminal_state": "STRICT_VALID", "ranked_item_ids": ["a", "b", "c", "d", "e", "target"]}
    valid, rank, hr, ndcg, mrr = freeze.utility(record, "target")
    assert (valid, rank, hr, ndcg) == (True, 5, 0.0, 0.0)
    assert mrr == pytest.approx(1 / 6)


def test_invalid_terminal_is_zero_even_with_target():
    record = {"strict_parse_valid": False, "terminal_state": "INVALID_LENGTH", "ranked_item_ids": ["target"]}
    assert freeze.utility(record, "target") == (False, None, 0.0, 0.0, 0.0)


def test_constant_paired_delta_ci():
    assert freeze.percentile_ci([0.25] * 20) == (0.25, 0.25, 1)
    assert freeze.percentile_ci([0.0] * 20) == (0.0, 0.0, 1)


def test_integrity_gate_fails_closed():
    with pytest.raises(ValueError, match="mismatch"):
        require(False, "mismatch")


def test_empty_bootstrap_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        freeze.percentile_ci([])


def test_packaged_config_preserves_frozen_protocol():
    config = freeze.configure(Path(__file__).resolve().parents[1], Path("configs/rq3_completion_freeze.yaml"))
    assert freeze.BOOTSTRAP_RESAMPLES == 10000
    assert freeze.BOOTSTRAP_SEED == 20260901
    assert config["experiment"]["allow_api_calls"] is False
    assert sum(freeze.EXPECTED_BY_H.values()) == 40794
