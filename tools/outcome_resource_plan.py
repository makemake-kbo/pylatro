"""Validate an explicit CPU-confirmation allocation without changing a pinned study."""

from pathlib import Path

from pylatro_agent.learning_bottleneck import read_json
from pylatro_agent.transfer import sha256

RESOURCE_PLAN = "resource_allocation_v2.json"


def cpu_confirmation_plan(root, path=None):
    root = Path(root)
    path = Path(path) if path is not None else root / RESOURCE_PLAN
    plan = read_json(path)
    expected = dict(
        version="cpu_confirmation_allocation_v1",
        study_manifest_sha256=sha256(root / "manifest.json"),
        accelerator_limit_seconds=read_json(root / "compute.json")["limit_seconds"],
        confirmation_device="cpu", confirmation_games=400,
        sampling_seeds=[0, 1, 2], training_replicates=[0, 1], parallel_modes=4,
    )
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("The CPU confirmation allocation does not match the pinned study")
    reserve = plan["remaining_accelerator_reserve_seconds"]
    original = read_json(root / "manifest.json")["confirmation_reserve_seconds"]
    if not 300 <= reserve <= original:
        raise ValueError("Retain at least five accelerator minutes for backend verification")
    evidence = plan["cpu_scheduling_verification"]
    if sha256(evidence["path"]) != evidence["sha256"] or not read_json(evidence["path"])["passed"]:
        raise ValueError("CPU scheduling verification changed or did not pass")
    return plan, dict(path=str(path), sha256=sha256(path))


def training_reserve(root, plan_path=None):
    root = Path(root)
    if plan_path is None:
        return read_json(root / "manifest.json")["confirmation_reserve_seconds"], None
    if (root / "candidate_selection.json").exists() or list(
            (root / "evaluations").glob("*_confirmation400/*/manifest.json")):
        raise ValueError("Do not extend training after confirmation selection or evaluation")
    plan, evidence = cpu_confirmation_plan(root, plan_path)
    return plan["remaining_accelerator_reserve_seconds"], evidence
