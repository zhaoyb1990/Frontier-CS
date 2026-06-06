"""TDD: adapter discovery + generation smoke test against the real research dir.

Uses the in-repo research/ tree (no Docker). Exercises nbody_simulation/random_10k
(CPU, gcc:13, no datasets) as the canonical P0 case.
"""
import json
from pathlib import Path

import pytest

from frontier_cs_research.adapter import FrontierCSResearchAdapter, discover_problems

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH = REPO_ROOT / "research"
NBODY = "nbody_simulation/random_10k"

pytestmark = pytest.mark.skipif(
    not (RESEARCH / "problems" / "nbody_simulation").is_dir(),
    reason="research/ tree not present",
)


def test_discover_finds_nbody_and_excludes_common_resources():
    problems = discover_problems(REPO_ROOT)
    ids = {p.problem_id for p in problems}
    assert NBODY in ids
    # 'common'/'resources' subdirs are scaffolding, never standalone problems
    assert not any("/common" in i or "/resources" in i or i.endswith("common") for i in ids)


def test_poc_generation_enumerated_per_cve_case():
    # We match the SCORED set (`frontier batch`'s _get_valid_problems), which
    # recurses into each per-CVE case — NOT the 4-category `frontier list` view.
    ids = {p.problem_id for p in discover_problems(REPO_ROOT)}
    poc = {i for i in ids if i.startswith("poc_generation")}
    # categories themselves have no evaluator.py, so they are NOT problems
    assert "poc_generation/heap_buffer_overflow" not in poc
    # the per-CVE cases ARE problems
    assert "poc_generation/heap_buffer_overflow/arvo_21000" in poc
    assert all(len(i.split("/")) == 3 for i in poc)  # family/category/case


def test_total_count_matches_batch_valid_problems():
    # Parity with the set the leaderboard CSVs average over (per-CVE poc):
    # equals `find research/problems -name evaluator.py` minus common/resources.
    problems_dir = REPO_ROOT / "research" / "problems"
    expected = {
        str(p.parent.relative_to(problems_dir))
        for p in problems_dir.rglob("evaluator.py")
        if "common" not in p.parts and "resources" not in p.parts
    }
    got = {p.problem_id for p in discover_problems(REPO_ROOT)}
    assert got == expected
    assert len(got) == 127


def test_nbody_problem_metadata():
    problems = {p.problem_id: p for p in discover_problems(REPO_ROOT)}
    p = problems[NBODY]
    assert p.ext == "cpp"
    assert p.language == "cpp"
    assert p.gpu is False
    assert p.docker_image == "gcc:13"
    assert p.task_id == "frontier-cs-research-nbody-simulation-random-10k"


def test_generate_nbody_task(tmp_path):
    adapter = FrontierCSResearchAdapter(REPO_ROOT, tmp_path, task_ids=[NBODY], overwrite=True)
    out = adapter.run()
    assert len(out) == 1
    task = out[0]
    assert task.name == "frontier-cs-research-nbody-simulation-random-10k"

    # core files exist
    assert (task / "task.toml").exists()
    assert (task / "instruction.md").exists()
    assert (task / "environment" / "Dockerfile").exists()
    assert (task / "solution" / "solve.sh").exists()
    assert (task / "tests" / "test.sh").exists()
    assert (task / "tests" / "evaluate.py").exists()

    # oracle reference is the cpp reference, baked into solution/
    assert (task / "solution" / "reference.cpp").exists()

    # baked problem subtree mirrors research_docker workspace (problem + sibling common)
    assert (task / "tests" / "research" / "nbody_simulation" / "random_10k" / "evaluator.py").exists()
    assert (task / "tests" / "research" / "nbody_simulation" / "common" / "evaluator_common.py").exists()
    assert (task / "tests" / "research" / "nbody_simulation" / "random_10k" / "evaluate.sh").exists()

    # problem_config.json carries ext/gpu for the verifier
    cfg = json.loads((task / "tests" / "problem_config.json").read_text())
    assert cfg["ext"] == "cpp"
    assert cfg["problem_id"] == NBODY
    assert cfg["gpu"] is False

    # CPU task -> gpus = 0 in task.toml; Dockerfile FROM gcc:13
    toml_text = (task / "task.toml").read_text()
    assert "gpus = 0" in toml_text
    assert "gcc:13" in (task / "environment" / "Dockerfile").read_text()


def test_generate_with_image_override(tmp_path):
    # China-friendly: base image can be redirected to a mirror registry
    mirror = "registry.example.com/mirror/gcc:13"
    adapter = FrontierCSResearchAdapter(
        REPO_ROOT, tmp_path, task_ids=[NBODY], overwrite=True, docker_image=mirror
    )
    task = adapter.run()[0]
    assert mirror in (task / "environment" / "Dockerfile").read_text()


def test_default_has_no_agent_layer(tmp_path):
    adapter = FrontierCSResearchAdapter(REPO_ROOT, tmp_path, task_ids=[NBODY], overwrite=True)
    task = adapter.run()[0]
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "claude-code" not in dockerfile
    assert "codex" not in dockerfile


VECTOR_ADD = "vector_addition/2_20"


def test_gpu_task_default_keeps_gpus_1_no_compose(tmp_path):
    adapter = FrontierCSResearchAdapter(REPO_ROOT, tmp_path, task_ids=[VECTOR_ADD], overwrite=True)
    task = adapter.run()[0]
    assert "gpus = 1" in (task / "task.toml").read_text()
    assert not (task / "environment" / "docker-compose.yaml").exists()


def test_local_gpu_emits_compose_and_gpus_0(tmp_path):
    adapter = FrontierCSResearchAdapter(
        REPO_ROOT, tmp_path, task_ids=[VECTOR_ADD], overwrite=True, local_gpu=True
    )
    task = adapter.run()[0]
    # harbor's local Docker validation needs gpus=0; GPU comes from the compose
    assert "gpus = 0" in (task / "task.toml").read_text()
    compose = (task / "environment" / "docker-compose.yaml")
    assert compose.exists()
    text = compose.read_text()
    assert "driver: nvidia" in text and "capabilities: [gpu]" in text


def test_local_gpu_does_not_touch_cpu_tasks(tmp_path):
    # CPU task: no compose, gpus already 0, --local-gpu is a no-op for it
    adapter = FrontierCSResearchAdapter(
        REPO_ROOT, tmp_path, task_ids=[NBODY], overwrite=True, local_gpu=True
    )
    task = adapter.run()[0]
    assert "gpus = 0" in (task / "task.toml").read_text()
    assert not (task / "environment" / "docker-compose.yaml").exists()


def test_with_agents_installs_cc_and_codex_via_npm(tmp_path):
    # both agents via npm, as the user runs them
    adapter = FrontierCSResearchAdapter(
        REPO_ROOT, tmp_path, task_ids=[NBODY], overwrite=True, with_agents=True
    )
    task = adapter.run()[0]
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "npm install -g" in dockerfile
    assert "@anthropic-ai/claude-code" in dockerfile
    assert "@openai/codex" in dockerfile
