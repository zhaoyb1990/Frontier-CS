"""TDD: pure metadata helpers (no Docker, no harbor)."""
from pathlib import Path

import pytest

from frontier_cs_research.utils import (
    normalize_task_id,
    detect_language,
    detect_gpu,
    find_reference,
    load_problem_config,
)


# ---- load_problem_config (tolerant) ----------------------------------------

def test_load_config_parses_yaml_and_json(tmp_path):
    y = tmp_path / "y.yaml"; y.write_text("tag: hpc\nruntime:\n  docker:\n    gpu: true\n")
    assert load_problem_config(y)["runtime"]["docker"]["gpu"] is True
    j = tmp_path / "j.yaml"; j.write_text('{"tag": "security"}')
    assert load_problem_config(j)["tag"] == "security"


def test_load_config_tolerates_malformed_yaml(tmp_path):
    # mirrors the real poc_generation/heap_use_after_free breakage:
    # a YAML scalar followed by a JSON block -> not valid YAML
    bad = tmp_path / "bad.yaml"
    bad.write_text('tag: security\n{\n  "dependencies": {"uv_project": "resources"}\n}\n')
    assert load_problem_config(bad) == {}  # falls back, does not raise


# ---- normalize_task_id -----------------------------------------------------

@pytest.mark.parametrize("pid,expected", [
    ("flash_attn", "frontier-cs-research-flash-attn"),
    ("nbody_simulation/random_10k", "frontier-cs-research-nbody-simulation-random-10k"),
    ("gemm_optimization/squares", "frontier-cs-research-gemm-optimization-squares"),
])
def test_normalize_task_id(pid, expected):
    assert normalize_task_id(pid) == expected


# ---- detect_gpu ------------------------------------------------------------

@pytest.mark.parametrize("cfg,expected", [
    ({"runtime": {"docker": {"gpu": True}}}, True),
    ({"runtime": {"resources": {"accelerators": "L4:1"}}}, True),
    ({"runtime": {"requires_gpu": True}}, True),
    ({"runtime": {"docker": {"image": "gcc:13"}}}, False),
    ({}, False),
])
def test_detect_gpu(cfg, expected):
    assert detect_gpu(cfg) is expected


# ---- detect_language -------------------------------------------------------

def test_detect_language_explicit_cpp(tmp_path):
    assert detect_language(tmp_path, {"runtime": {"language": "cpp"}}) == ("cpp", "cpp")


def test_detect_language_explicit_python(tmp_path):
    assert detect_language(tmp_path, {"runtime": {"language": "python"}}) == ("python", "py")


def test_detect_language_infer_cpp_from_reference(tmp_path):
    (tmp_path / "reference.cpp").write_text("// x")
    assert detect_language(tmp_path, {}) == ("cpp", "cpp")


def test_detect_language_default_python(tmp_path):
    assert detect_language(tmp_path, {}) == ("python", "py")


# ---- find_reference --------------------------------------------------------

def test_find_reference_prefers_problem_dir(tmp_path):
    prob = tmp_path / "problems" / "p"
    prob.mkdir(parents=True)
    ref = prob / "reference.cpp"
    ref.write_text("// ref")
    sols = tmp_path / "solutions"
    assert find_reference(prob, sols, "p", "cpp") == ref


def test_find_reference_falls_back_to_solutions(tmp_path):
    prob = tmp_path / "problems" / "p"
    prob.mkdir(parents=True)
    sols = tmp_path / "solutions" / "p"
    sols.mkdir(parents=True)
    cand = sols / "gpt5.py"
    cand.write_text("# sol")
    got = find_reference(prob, tmp_path / "solutions", "p", "py")
    assert got == cand


def test_find_reference_none_when_absent(tmp_path):
    prob = tmp_path / "problems" / "p"
    prob.mkdir(parents=True)
    assert find_reference(prob, tmp_path / "solutions", "p", "py") is None
