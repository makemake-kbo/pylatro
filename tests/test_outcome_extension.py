import json
import math

import pytest

from tools.extend_outcome_pilots import fitting_extension, verify_predecessor_reports


def test_extension_fits_every_matched_leg_and_retains_admission_margin():
    costs = dict(control0=190, treatment0=240, control1=200, treatment1=250)
    updates, hours = fitting_extension(costs, remaining=14000, reserve=300, maximum_updates=15)
    assert updates == 13
    assert list(hours) == list(costs)
    remaining = 14000
    for name, limit in hours.items():
        seconds = 3600 * limit
        assert seconds == pytest.approx(1.1 * costs[name] * updates)
        assert seconds + 600 + 300 <= remaining
        assert 1.2 * seconds + 5 <= remaining
        remaining -= seconds
    assert remaining >= 300
    assert fitting_extension(costs, remaining=3000, reserve=300, maximum_updates=15) == (0, {})
    assert fitting_extension(costs, remaining=1e6, reserve=300, maximum_updates=15)[0] == 15


def test_extension_rejects_nonfinite_or_unmeasured_costs():
    for costs in ({}, {"a": 0}, {"a": -1}, {"a": math.nan}, {"a": math.inf}):
        with pytest.raises(ValueError, match="Measured positive"):
            fitting_extension(costs, remaining=10000, reserve=300, maximum_updates=15)


def test_predecessor_reports_accept_serialized_training_seed_keys_but_reject_changed_results(tmp_path):
    computed = dict(results=dict(per_replicate={0: dict(wins=174), 1: dict(wins=164)}),
                    evidence=dict(seed0=dict(sha256="unchanged")))
    path = tmp_path / "comparison.json"
    original = json.dumps(computed)
    path.write_text(original)
    verify_predecessor_reports([path], [computed])
    assert path.read_text() == original
    changed = json.loads(original)
    changed["results"]["per_replicate"]["0"]["wins"] = 175
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from verified"):
        verify_predecessor_reports([path], [computed])
