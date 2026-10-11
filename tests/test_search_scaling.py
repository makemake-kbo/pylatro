from __future__ import annotations

import gzip
import importlib.util
import json
import signal
from pathlib import Path

import numpy as np
import pytest
from test_public_hybrid import hand_state, private_mutation, snapshot

from pylatro_agent.hybrid import HybridPolicy, Plan, PlannerConfig, PlanningDeadline
from pylatro_agent.public_state import simulation_state
from pylatro_agent.search_diagnostics import historical_groups, inspect_trace, public_resources

spec = importlib.util.spec_from_file_location(
    "search_scaling_study", Path(__file__).parents[1] / "tools/search_scaling_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def plans_for(state):
    from pylatro_agent.constants import ActionRange as AR
    from pylatro_agent.subset_actions import subset_index

    return [Plan((int(AR.PLAY_SUBSET_START + subset_index(indices)),)) for indices in ((0,), (0, 1, 2, 3, 4))]


@pytest.mark.parametrize("count", [4, 16, 32, 64])
def test_shared_sample_prefix_and_equal_completion(monkeypatch, count):
    state = hand_state()
    plans = plans_for(state)
    policy = HybridPolicy(lambda s: plans[0].actions[0], state.data, PlannerConfig(samples=count))
    monkeypatch.setattr(policy, "_hand_plans", lambda *a: plans)
    seen = []

    def trial(s, p, i, hand):
        seen.append((i, p.actions))
        return ((i, 0, 0, 0), True)

    monkeypatch.setattr(policy, "_trial", trial)
    d = policy.decide(snapshot(state), trace=True)
    assert d.action == d.proposal
    assert seen == [(i, p.actions) for i in range(count) for p in plans]
    assert d.trace["sample_indices"] == list(range(count))
    assert len(d.trace["completed_scores"]) == count


@pytest.mark.parametrize("count", [4, 16, 32, 64])
@pytest.mark.parametrize("stop", ["deadline", "continuation_cap"])
def test_partial_round_discarded_with_larger_profiles(monkeypatch, count, stop):
    state = hand_state()
    plans = plans_for(state)
    policy = HybridPolicy(lambda s: plans[0].actions[0], state.data, PlannerConfig(samples=count))
    monkeypatch.setattr(policy, "_hand_plans", lambda *a: plans)

    def trial(s, p, i, hand):
        if i == 2 and p == plans[1]:
            if stop == "deadline":
                raise PlanningDeadline
            return ((100, 0, 0, 0), False)
        return ((100 if i == 2 else 0, 0, 0, 0), True)

    monkeypatch.setattr(policy, "_trial", trial)
    result = policy.decide(snapshot(state), trace=True)
    assert result.diagnostics["samples"] == 2
    assert result.action == result.proposal
    assert result.trace["stopping_reason"] == stop
    assert result.trace["completed_scores"] == [[(0, 0, 0, 0)] * 2] * 2


def test_configuration_cache_and_independent_batch():
    state = hand_state()
    p = plans_for(state)[0]
    policy = HybridPolicy(lambda s: p.actions[0], state.data)
    public = snapshot(state)
    for count in (4, 16, 32, 64):
        policy.config = PlannerConfig(samples=count)
        policy._trial(public, p, 0, True)
    assert len(policy.cache) == 4
    policy.config = PlannerConfig(sample_start=10000)
    policy._trial(public, p, 10000, True)
    assert len(policy.cache) == 5
    assert simulation_state(public, state.data, 0).seed != simulation_state(public, state.data, 10000).seed


def test_trace_is_public_optional_and_outside_timer(monkeypatch):
    state = hand_state()
    public = snapshot(state)
    proposal = plans_for(state)[0].actions[0]
    policy = HybridPolicy(lambda s: proposal, state.data, PlannerConfig(mode="hand", samples=16, sample_start=10000))
    monkeypatch.setattr(policy, "_hand_plans", lambda *a: plans_for(state))
    monkeypatch.setattr(policy, "_trial", lambda *a: ((0, 0, 0, 0), True))
    original = policy._trace

    def check(*args):
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
        return original(*args)

    monkeypatch.setattr(policy, "_trace", check)
    plain = policy.decide(public)
    traced = policy.decide(snapshot(private_mutation(state)), trace=True)
    assert plain.trace is None
    assert traced.action == plain.action
    assert traced.trace["snapshot"] == public.values
    assert traced.trace["sample_indices"] == list(range(10000, 10016))
    encoded = json.dumps(traced.trace)
    for private in ("pseudorandom", "reward_uid", "draw_pile", "PRIVATE_DIFFERENT", '"seed"'):
        assert private not in encoded
    assert json.loads(encoded)["completed_scores"]


def test_default_unchanged_and_config_validation():
    assert PlannerConfig().samples == 4 and PlannerConfig().seconds == 3
    for count in (0, 1, 8, 4.0, True):
        with pytest.raises(ValueError):
            PlannerConfig(samples=count)
    for seconds in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            PlannerConfig(seconds=seconds)
    with pytest.raises(ValueError):
        PlannerConfig(sample_start=-1)


def rows(wins, arm="incumbent"):
    return [
        dict(
            seed=f"seed-{i}",
            won=w,
            complete=True,
            decision_seconds=[1.0],
            steps=1,
            changes=0,
            searches=0,
            fallbacks=0,
            paired_sample_sets=0,
            seconds=1,
            ante=8 if w else 3,
            search_stops={},
            sample_counts={},
            boundary_audits=1,
        )
        for i, w in enumerate(wins)
    ]


def test_runtime_selection_ignores_wins_and_preserves_fixed_panels():
    timings = {
        a: [dict(seconds=t, won=False) for _ in range(8)] for a, t in zip(study.PROFILES, [10, 20, 40, 80], strict=True)
    }
    result = study.runtime_selection(timings, 100000)
    assert result["high"] == "high64"
    for values in timings.values():
        for r in values:
            r["won"] = True
    assert result == study.runtime_selection(timings, 100000)
    assert study.runtime_selection(timings, result["projections"]["high64"]["wall_with_margin"] - 1)["high"] == "high32"
    assert study.runtime_selection(timings, 1)["high"] is None
    expected_worker = 96 * (10 + 20 + 80) + 256 * (10 + 80)
    assert result["projections"]["high64"]["worker_seconds"] == expected_worker


def test_selection_gain_threshold_and_ties():
    panels = {a: rows([True] * 48 + [False] * 48) for a in ("incumbent", "moderate", "high64")}
    assert study.select_challenger(panels) is None
    panels["high64"] = rows([True] * 52 + [False] * 44)
    assert study.select_challenger(panels) is None
    panels["high64"] = rows([True] * 53 + [False] * 43)
    assert study.select_challenger(panels) == "high64"
    panels["moderate"] = rows([True] * 53 + [False] * 43)
    assert study.select_challenger(panels) == "moderate"
    panels["high64"][0]["decision_seconds"] = [0.5]
    assert study.select_challenger(panels) == "high64"


def test_adoption_requires_all_pairs_positive_interval_and_five_points():
    b = rows([True] * 128 + [False] * 128)
    c = rows([True] * 141 + [False] * 115)
    assert study.adoption(b, c)[0]
    assert not study.adoption(b, rows([True] * 140 + [False] * 116))[0]
    with pytest.raises(ValueError):
        study.adoption(b[:-1], c[:-1])
    with pytest.raises(ValueError):
        study.paired(b, c[::-1])
    assert study.paired(b, b)["interval95"] == [0, 0]
    assert study.paired(b, c) == study.paired(b, c)


def valid_row(identity):
    return dict(
        rows([False])[0],
        identity=identity,
        identity_sha256=study.fingerprint(identity),
        planner_config=identity["planner"],
        mode="hand",
        sample=False,
        illegal_actions=0,
        boundary_violations=0,
        interpretation="public-information-v2",
        finished_unix=10,
        seed=identity["seed"],
    )


def test_atomic_resume_rejects_mismatch_missing_duplicates_and_integrity(tmp_path):
    config = dict(checkpoint=dict(sha256="abc"))
    identity = study.job_identity("development", "incumbent", "seed-0", "frozen", config)
    path = tmp_path / "development/incumbent/seed-0.json"
    row = valid_row(identity)
    study.atomic(path, row)
    assert study.digest(path) == study.fingerprint(row)
    assert study.read_panel(tmp_path, "development", "incumbent", ["seed-0"], "frozen", config, 20) == [row]
    with pytest.raises(ValueError, match="Duplicate"):
        study.read_panel(tmp_path, "development", "incumbent", ["seed-0"] * 2, "frozen", config, 20)
    with pytest.raises(ValueError, match="Incomplete"):
        study.read_panel(tmp_path, "development", "incumbent", ["seed-0", "missing"], "frozen", config, 20)
    for field, value in (
        ("complete", False),
        ("illegal_actions", 1),
        ("boundary_violations", 1),
        ("finished_unix", 30),
        ("identity_sha256", "other"),
        ("won", 0),
    ):
        with pytest.raises(ValueError):
            study.validate_row(dict(row, **{field: value}), identity, 20)
    # An interrupted temporary record never replaces the committed game.
    path.with_suffix(".json.123.tmp").write_text('{"complete":')
    assert study.read(path) == row
    with pytest.raises(TypeError):
        study.atomic(path, dict(bad=object()))
    assert study.read(path) == row


def test_json_serialization_and_original_clock(tmp_path):
    x = json.loads(study.encode(dict(passed=np.bool_(True), n=np.int64(10), rate=np.float64(0.5))))
    assert x == dict(passed=True, n=10, rate=0.5)
    with pytest.raises(ValueError):
        study.encode(dict(rate=float("nan")))
    first = study.clock_start(tmp_path, now=100)
    assert study.clock_start(tmp_path, now=99999) == first
    assert first["deadline_unix"] == 86500
    study.atomic(tmp_path / "clock.json", dict(started_unix=100, deadline_unix=100 + 100000))
    with pytest.raises(ValueError):
        study.clock_start(tmp_path)


def test_seed_separation_and_metadata_boundary():
    panels = study.seed_panels()
    assert {k: len(v) for k, v in panels.items()} == dict(pilot=8, development=96, confirmation=256)
    study.assert_separation(panels, [str(i) for i in range(1000)])
    with pytest.raises(ValueError):
        study.assert_separation(panels, panels["pilot"])
    assert study.metadata_name("reserved_seed_manifest.json")
    assert not study.metadata_name("reserved_outcomes.jsonl")
    assert not study.metadata_name("seed_results.json")


def test_fixed_diagnostic_strata_and_replay_limits():
    b = rows([True] * 32 + [False] * 115 + [False] * 85 + [True] * 168)
    h = rows([False] * 32 + [False] * 115 + [True] * 85 + [True] * 168)
    result = historical_groups(b, h)
    assert result["counts"] == dict(regressions=32, shared_losses=115, rescues=85, shared_wins=168)
    assert {k: len(v) for k, v in result["diagnostic_subset"].items()} == dict(
        regressions=32, shared_losses=16, rescues=8, shared_wins=8
    )
    for panel in result["diagnostic_subset"].values():
        assert panel == sorted(panel, key=lambda s: study.hashlib.sha256(s.encode()).hexdigest())


def test_trace_reconstruction_and_resources_are_public(tmp_path):
    state = hand_state()
    public = snapshot(state)
    resources = public_resources(public, state.data)
    assert resources == public_resources(snapshot(private_mutation(state)), state.data)
    proposal = plans_for(state)[0].actions[0]
    d = HybridPolicy(lambda s: proposal, state.data, PlannerConfig(seconds=0)).decide(public, trace=True)
    row = dict(
        step=0,
        **d.trace,
        public_resources=resources,
        observed_transition=dict(resources=resources, snapshot=public.values, done=False, won=False),
    )
    path = tmp_path / "trace.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(row) + "\n")
    info = inspect_trace(path)
    assert info["first_eligible"]["proposal"] == proposal
    assert info["entry"] == resources
    assert info["decisions"] == 1


def test_real_public_world_score_prefixes(monkeypatch):
    state = hand_state()
    state.hand_cards[0].center_key = "m_lucky"
    plans = plans_for(state)
    scores = []
    for count in (4, 16, 32, 64):
        policy = HybridPolicy(lambda s: plans[0].actions[0], state.data, PlannerConfig(samples=count, seconds=10))
        monkeypatch.setattr(policy, "_hand_plans", lambda *a: plans)
        d = policy.decide(snapshot(state), trace=True)
        assert len(d.trace["completed_scores"]) == count
        scores.append(d.trace["completed_scores"])
    for count, values in zip((4, 16, 32, 64), scores, strict=True):
        assert values == scores[-1][:count]


def test_search_does_not_advance_policy_or_global_rng(monkeypatch):
    import torch

    from pylatro_agent.public_runtime import PublicNeuralPolicy

    checkpoint = Path(__file__).parents[1] / "incumbents/ppo_replica1/checkpoint.pt"
    state = hand_state()
    public = snapshot(state)
    base = PublicNeuralPolicy(checkpoint, state.data, sample=True, policy_seed=551)
    global_before = torch.get_rng_state().clone()
    expected = base(public)
    advanced = base.rng_state.clone()
    base.reset_rng(551)
    policy = HybridPolicy(base, state.data, PlannerConfig(samples=64, mode="hand"))
    monkeypatch.setattr(policy, "_hand_plans", lambda s, p: [Plan((p,))])
    monkeypatch.setattr(policy, "_trial", lambda *a: ((0, 0, 0, 0), True))
    d = policy.decide(public, trace=True)
    assert d.proposal == expected
    assert torch.equal(base.rng_state, advanced)
    assert torch.equal(torch.get_rng_state(), global_before)


def test_worker_seals_pilot_and_commits_compressed_diagnostic(monkeypatch, tmp_path):
    import pylatro_agent.hybrid_game as game

    config = dict(checkpoint=dict(sha256="abc"))

    def fake_game(checkpoint, seed, **kwargs):
        identity = study.job_identity(stage, "incumbent", seed, "frozen", config)
        if kwargs["trace_sink"] is not None:
            kwargs["trace_sink"](dict(public="visible"))
        row = valid_row(identity)
        for k in ("identity", "identity_sha256", "finished_unix"):
            row.pop(k)
        return row

    monkeypatch.setattr(game, "play_game", fake_game)
    for stage in ("pilot", "diagnostic"):
        identity = study.job_identity(stage, "incumbent", "seed-0", "frozen", config)
        path = tmp_path / stage / "seed-0.json"
        assert study.worker(dict(identity=identity, path=str(path), checkpoint="unused", deadline=10**20))["complete"]
        result = study.read(path)
        assert result["won"] is False
        if stage == "pilot":
            timing = study.read(path.with_suffix(".timing.json"))
            assert "won" not in timing and "ante" not in timing
            assert timing["outcome_sha256"] == study.digest(path)
        else:
            with gzip.open(path.with_suffix(".trace.jsonl.gz"), "rt") as f:
                assert json.loads(f.readline()) == dict(public="visible")
            assert result["trace_sha256"] == study.digest(path.with_suffix(".trace.jsonl.gz"))
