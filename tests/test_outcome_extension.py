import math

import pytest

from tools.extend_outcome_pilots import fitting_extension


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
