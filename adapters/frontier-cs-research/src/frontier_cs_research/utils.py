from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchProblem:
    """Parsed metadata for a Frontier-CS research problem."""

    problem_id: str          # rel path under research/problems, e.g. "nbody_simulation/random_10k"
    task_id: str             # normalized harbor task id
    problem_dir: Path
    statement: str
    tag: str
    language: str            # "python" | "cpp"
    ext: str                 # "py" | "cpp"
    gpu: bool
    accelerators: str | None
    uv_project: str | None
    timeout_seconds: int
    docker_image: str
    reference_path: Path | None
    config: dict[str, Any]


def load_problem_config(config_path: Path) -> dict[str, Any]:
    """Load config.yaml. yaml.safe_load also parses the JSON-style configs
    (e.g. qknorm) since JSON is a subset of YAML.

    Tolerant by design: a few upstream config files are malformed (e.g.
    poc_generation/heap_use_after_free mixes a YAML scalar with a JSON block).
    A single bad file must not break discovery of every other problem, so we
    fall back to an empty config (→ defaults) instead of raising."""
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def normalize_task_id(problem_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", problem_id).strip("-").lower()
    return f"frontier-cs-research-{slug}"


def read_problem_statement(problem_dir: Path) -> str:
    for name in ("readme", "statement.txt", "README.md", "readme.md"):
        path = problem_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def detect_language(problem_dir: Path, config: dict[str, Any]) -> tuple[str, str]:
    """Return (language, ext). Explicit runtime.language wins; otherwise infer
    from a reference file in the problem dir; default to python."""
    runtime = config.get("runtime", {}) or {}
    lang = runtime.get("language")
    if lang:
        lang = str(lang).lower()
        return ("cpp", "cpp") if lang in ("cpp", "c++", "cxx") else (lang, "py")
    if (problem_dir / "reference.cpp").exists():
        return ("cpp", "cpp")
    if (problem_dir / "reference.py").exists():
        return ("python", "py")
    return ("python", "py")


def detect_gpu(config: dict[str, Any]) -> bool:
    runtime = config.get("runtime", {}) or {}
    docker = runtime.get("docker", {}) or {}
    resources = runtime.get("resources", {}) or {}
    return bool(
        docker.get("gpu")
        or runtime.get("requires_gpu")
        or resources.get("accelerators")
    )


# research/solutions/<id>/ holds model-generated solutions of varying quality
# (and some that don't even run). Prefer the strongest models' base variant so
# the oracle isn't an arbitrary alphabetical pick (which lands on a broken
# deepseekreasoner_1.py for several kernel problems).
PREFERRED_MODELS = (
    "gpt5_high", "gpt5.2", "gpt5.1", "gpt5", "gpt5_medium",
    "gemini3pro", "gemini2.5pro", "trinitylargethinking",
    "deepseekreasoner", "grok4fastreasoning",
)


def find_reference(
    problem_dir: Path, solutions_root: Path, problem_id: str, ext: str
) -> Path | None:
    """Pick the oracle reference solution.

    Preference: problem_dir/reference.<ext> > solutions/<id>/reference*.<ext> >
    the base variant of the strongest available model > first solutions/<id>/*.<ext>.
    These are model outputs, not hand-tuned optima, so even the chosen one may
    score 0 on hard kernel problems — that's the benchmark, not a bug.
    """
    p = problem_dir / f"reference.{ext}"
    if p.exists():
        return p
    sdir = solutions_root / problem_id
    if sdir.is_dir():
        for cand in (f"reference.{ext}", f"reference_baseline.{ext}"):
            if (sdir / cand).exists():
                return sdir / cand
        for model in PREFERRED_MODELS:
            cand = sdir / f"{model}.{ext}"  # base variant (no _1/_2 suffix)
            if cand.exists():
                return cand
        files = sorted(sdir.glob(f"*.{ext}"))
        if files:
            return files[0]
    return None
