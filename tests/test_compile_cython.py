from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import Extension

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compile_cython.py"


def _load_compile_cython_module():
    spec = importlib.util.spec_from_file_location("compile_cython_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_distribution_uses_src_as_package_root() -> None:
    module = _load_compile_cython_module()
    dist = module._build_distribution([Extension("pylatro.rng", sources=["dummy.c"])])
    cmd = dist.get_command_obj("build_ext")
    cmd.ensure_finalized()
    cmd.inplace = True

    fullpath = Path(cmd.get_ext_fullpath("pylatro.rng")).resolve()
    assert fullpath.parent == module.SRC.resolve()


def test_clean_extensions_removes_legacy_nested_artifacts(tmp_path, monkeypatch) -> None:
    module = _load_compile_cython_module()
    src = tmp_path / "src" / "pylatro"
    src.mkdir(parents=True)
    src_agent = tmp_path / "src" / "pylatro_agent"
    src_agent.mkdir(parents=True)
    legacy_src = src / "pylatro"
    legacy_src.mkdir()

    (src / "rng.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    (src / "rng.c").write_text("/* generated */", encoding="utf-8")
    (legacy_src / "rng.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SRC", src)
    monkeypatch.setattr(module, "SRC_AGENT", src_agent)

    module.clean_extensions()

    assert not (src / "rng.cpython-312-x86_64-linux-gnu.so").exists()
    assert not (src / "rng.c").exists()
    assert not legacy_src.exists()
    assert not build_dir.exists()
