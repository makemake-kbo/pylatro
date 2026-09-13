import pytest

from pylatro_agent.constants import SubPhase
from pylatro_agent.phase_oracle import policy_seed, teacher_controls_phase


@pytest.mark.parametrize(
    "arm,phases",
    [
        ("student", set()),
        ("hand", {SubPhase.CHOOSE_ACTION}),
        ("shop", {SubPhase.SHOP}),
        ("hand_shop", {SubPhase.CHOOSE_ACTION, SubPhase.SHOP}),
        ("teacher_all", set(SubPhase)),
    ],
)
def test_phase_replacement_boundaries(arm, phases):
    for phase in SubPhase:
        assert teacher_controls_phase(arm, phase) == (phase in phases)


def test_phase_oracle_rejects_unknown_arm_and_uses_independent_game_rng():
    with pytest.raises(ValueError, match="Unknown"):
        teacher_controls_phase("accidental", SubPhase.SHOP)
    seeds = [policy_seed(game, repeat) for game in (5000000, 5000001) for repeat in (0, 1, 2)]
    assert len(set(seeds)) == 6
    assert policy_seed(5000000, 0) == seeds[0]
