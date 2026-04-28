from math import isclose

from pylatro.rng import PseudorandomState, pseudohash


def test_pseudohash_and_pseudoseed_match_reference_fixture() -> None:
    state = PseudorandomState("AAAAAAAA")

    assert isclose(pseudohash("AAAAAAAA"), 0.43257138351543745)
    assert isclose(state.pseudoseed("boss"), 0.5374444774613187)
    assert isclose(state.pseudoseed("Voucher1"), 0.5290234269488188)
    assert isclose(state.pseudoseed("Tag1"), 0.2909399952510187)
    assert isclose(state.pseudoseed("shuffle"), 0.6564266916414188)
    assert isclose(state.pseudoseed("rarity1"), 0.4462461409768187)


def test_repeated_pseudoseed_progresses_state() -> None:
    state = PseudorandomState("AAAAAAAA")
    first = state.pseudoseed("boss")
    second = state.pseudoseed("boss")
    third = state.pseudoseed("boss")

    assert isclose(first, 0.5374444774613187)
    assert isclose(second, 0.3372904636163687)
    assert isclose(third, 0.49216242764296875)
    assert state.random_string(8, second) == "CQ5IQGBQ"
    assert state.random_string(8, third) == "NYVU6UGQ"

