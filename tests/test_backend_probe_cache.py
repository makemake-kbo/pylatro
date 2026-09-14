import json
import sys

import pytest
import torch

from pylatro_agent.transfer import sha256
from tools.check_subset_backend import main


def test_backend_probe_rejects_changed_cache_before_model_or_accelerator_loading(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(development_seeds=[1, 2, 3, 4])))
    probe = tmp_path / "probe.pt"
    metadata = probe.with_suffix(".json")
    probe.write_bytes(b"This is deliberately not a checkpoint")
    metadata.write_text(json.dumps(dict(cache_sha256="0" * 64)))
    monkeypatch.setattr(sys, "argv", [
        "check_subset_backend.py", "--root", str(tmp_path), "--checkpoint", str(tmp_path / "absent.pt"),
        "--observation-probe", str(probe),
    ])
    with pytest.raises(ValueError, match="observation cache changed"):
        main()
    torch.save(dict(metadata=dict(source_core_sha256="changed")), probe)
    metadata.write_text(json.dumps(dict(source_core_sha256="original", cache_sha256=sha256(probe))))
    with pytest.raises(ValueError, match="metadata differs"):
        main()
