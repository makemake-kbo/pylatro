import pytest

from tools.audit_outcome_labels import label_metrics, reconstruct_labels


def test_long_wins_can_lose_initial_outcome_labels_while_short_losses_keep_them():
    rows = [dict(environment=1, steps=100, update=1, won=False),
            dict(environment=0, steps=300, update=2, won=True),
            dict(environment=1, steps=200, update=2, won=False)]
    result = reconstruct_labels(rows, start_update=0, rollout_length=256)
    assert [row["label_available_states"] for row in result] == [100, 44, 44]
    assert [row["initial_state_label_available"] for row in result] == [True, False, False]
    metrics = label_metrics(result)
    assert metrics["observed_win_rate"] == 1 / 3
    assert metrics["win_rate_in_labeled_initial_states"] == 0
    resumed = [dict(row, update=row["update"] + 20) for row in rows]
    assert reconstruct_labels(resumed, 20, 256)[1]["label_available_states"] == 44
    with pytest.raises(ValueError, match="recorded rollout boundary"):
        reconstruct_labels(resumed, 0, 256)
