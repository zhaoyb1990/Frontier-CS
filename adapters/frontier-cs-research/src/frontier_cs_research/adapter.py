from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from harbor.models.task.paths import TaskPaths

from .utils import (
    ResearchProblem,
    detect_gpu,
    detect_language,
    find_reference,
    load_problem_config,
    normalize_task_id,
    read_problem_statement,
)

LOGGER = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "task-template"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")

# Optional agent layer for `harbor trial -a claude-code|codex -m <model>`.
# Both CLIs are installed via npm (claude-code's npm package is
# @anthropic-ai/claude-code). Skip this entirely (default) when the base image
# already bundles the agents, or for the oracle-only / offline path.
AGENT_LAYER = r"""# Agent CLIs (claude-code + codex) for model-driven `harbor trial` runs.
# Both via npm. China: `npm config set registry https://registry.npmmirror.com`
# before building, or bake the agents into the base image and drop --with-agents.
ENV NVM_DIR="/root/.nvm"
RUN if ! command -v npm >/dev/null 2>&1; then \
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
        . "$NVM_DIR/nvm.sh" && nvm install 22 && nvm alias default 22 && \
        ln -sf "$(which node)" /usr/local/bin/node && ln -sf "$(which npm)" /usr/local/bin/npm; \
    fi && \
    npm install -g @anthropic-ai/claude-code @openai/codex && \
    (ln -sf "$(command -v claude)" /usr/local/bin/claude 2>/dev/null || true) && \
    (ln -sf "$(command -v codex)" /usr/local/bin/codex 2>/dev/null || true)
ENV CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
"""


def _make_task_paths(task_dir: Path):
    try:
        from harbor.models.task.paths import TaskPaths

        return TaskPaths(task_dir=task_dir)
    except ImportError:
        return SimpleNamespace(
            task_dir=task_dir,
            environment_dir=task_dir / "environment",
            solution_dir=task_dir / "solution",
            tests_dir=task_dir / "tests",
            instruction_path=task_dir / "instruction.md",
            config_path=task_dir / "task.toml",
        )


def _int_from(value, default: int) -> int:
    if value is None:
        return default
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else default


def _build_problem(rel_id: str, problem_dir: Path, solutions_root: Path) -> ResearchProblem:
    config_path = problem_dir / "config.yaml"
    config = load_problem_config(config_path) if config_path.exists() else {}
    runtime = config.get("runtime", {}) or {}
    docker = runtime.get("docker", {}) or {}
    resources = runtime.get("resources", {}) or {}
    dependencies = config.get("dependencies", {}) or {}
    language, ext = detect_language(problem_dir, config)
    return ResearchProblem(
        problem_id=rel_id,
        task_id=normalize_task_id(rel_id),
        problem_dir=problem_dir,
        statement=read_problem_statement(problem_dir),
        tag=str(config.get("tag", "research")),
        language=language,
        ext=ext,
        gpu=detect_gpu(config),
        accelerators=resources.get("accelerators"),
        uv_project=dependencies.get("uv_project"),
        timeout_seconds=_int_from(runtime.get("timeout_seconds"), 1800),
        docker_image=str(docker.get("image", "python:3.11-slim-trixie")),
        reference_path=find_reference(problem_dir, solutions_root, rel_id, ext),
        config=config,
    )


def discover_problems(frontier_cs_root: Path) -> list[ResearchProblem]:
    """Enumerate research problems the way the batch evaluator actually scores
    them: one task per evaluator.py (excluding shared common/resources dirs).

    This matches `frontier batch`'s `_get_valid_problems()` — the set used to
    compute the leaderboard CSVs — which recurses into poc_generation's per-CVE
    cases (e.g. poc_generation/heap_buffer_overflow/arvo_21000). Note the
    leaderboard *header* hardcodes "68 problems" and `frontier list research`
    collapses poc to 4 categories, but neither is the set the scores average
    over; the scored set is the per-case one (127 at this revision)."""
    problems_dir = frontier_cs_root / "research" / "problems"
    solutions_root = frontier_cs_root / "research" / "solutions"
    if not problems_dir.is_dir():
        raise FileNotFoundError(f"Frontier-CS research problems not found: {problems_dir}")

    problems: list[ResearchProblem] = []
    for evaluator_path in sorted(problems_dir.rglob("evaluator.py")):
        problem_dir = evaluator_path.parent
        rel_id = str(problem_dir.relative_to(problems_dir))
        parts = rel_id.split("/")
        # 'common'/'resources' are shared scaffolding, never standalone problems
        if "common" in parts or "resources" in parts:
            continue
        problems.append(_build_problem(rel_id, problem_dir, solutions_root))

    return problems


class FrontierCSResearchAdapter:
    """Generate Harbor tasks for Frontier-CS research problems."""

    def __init__(
        self,
        frontier_cs_root: Path,
        output_dir: Path,
        *,
        limit: int | None = None,
        overwrite: bool = False,
        task_ids: Iterable[str] | None = None,
        template_dir: Path | None = None,
        docker_image: str | None = None,
        with_agents: bool = False,
    ):
        self.root = Path(frontier_cs_root)
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = set(task_ids) if task_ids is not None else None
        self.template_dir = Path(template_dir or TEMPLATE_DIR)
        self.docker_image = docker_image
        self.with_agents = with_agents

    def run(self) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        problems = discover_problems(self.root)
        if self.task_ids is not None:
            problems = [p for p in problems if p.problem_id in self.task_ids]
        if self.limit is not None:
            problems = problems[: self.limit]

        results: list[Path] = []
        for problem in problems:
            generated = self.generate_task(problem, overwrite=self.overwrite)
            if generated is not None:
                results.append(generated)
        return results

    def generate_task(
        self, problem: ResearchProblem, *, overwrite: bool = False
    ) -> Path | None:
        task_dir = self.output_dir / problem.task_id
        if task_dir.exists():
            if not overwrite:
                LOGGER.info("Skipping %s (already exists)", task_dir.name)
                return None
            shutil.rmtree(task_dir)

        task_paths = _make_task_paths(task_dir)
        task_paths.task_dir.mkdir(parents=True, exist_ok=True)
        task_paths.environment_dir.mkdir(parents=True, exist_ok=True)
        task_paths.solution_dir.mkdir(parents=True, exist_ok=True)
        task_paths.tests_dir.mkdir(parents=True, exist_ok=True)

        self._write_instruction(task_paths, problem)
        self._write_environment(task_paths, problem)
        self._write_tests(task_paths, problem)
        self._write_solution(task_paths, problem)
        self._write_task_config(task_paths, problem)
        LOGGER.info("  [OK] %s (%s, gpu=%s)", problem.problem_id, problem.ext, problem.gpu)
        return task_paths.task_dir

    def _write_instruction(self, task_paths: "TaskPaths", problem: ResearchProblem) -> None:
        solution_path = f"/app/solution.{problem.ext}"
        header = (
            "You are solving a Frontier-CS research (systems/ML performance) problem.\n\n"
            f"Write your solution to `{solution_path}`. The problem readme below "
            "defines the required `Solution` interface (a `Solution` class with a "
            "`solve()` method whose signature depends on the problem) and the scoring "
            "formula. Only the final solution file is evaluated.\n\n"
            f"Problem id: `{problem.problem_id}`\n"
            f"Language: `{problem.language}`\n"
            f"GPU: `{'yes' if problem.gpu else 'no'}`"
            + (f" ({problem.accelerators})" if problem.accelerators else "")
            + "\n\n"
            "Original problem statement:\n\n"
        )
        task_paths.instruction_path.write_text(
            header + problem.statement.rstrip() + "\n", encoding="utf-8"
        )

    def _write_environment(self, task_paths: "TaskPaths", problem: ResearchProblem) -> None:
        env_dir = task_paths.environment_dir
        dockerfile = (self.template_dir / "environment" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        image = self.docker_image or problem.docker_image
        agent_layer = AGENT_LAYER if self.with_agents else ""
        env_dir.joinpath("Dockerfile").write_text(
            dockerfile.replace("{base_image}", image).replace(
                "{agent_layer}", agent_layer
            ),
            encoding="utf-8",
        )

    def _write_tests(self, task_paths: "TaskPaths", problem: ResearchProblem) -> None:
        tests_dir = task_paths.tests_dir
        shutil.copy2(self.template_dir / "tests" / "test.sh", tests_dir / "test.sh")
        shutil.copy2(self.template_dir / "tests" / "evaluate.py", tests_dir / "evaluate.py")
        (tests_dir / "test.sh").chmod(0o755)

        # Bake the problem subtree the way research_docker lays out its workspace:
        # research/<problem_id>/ plus sibling research/<parent>/common/ dirs.
        research_root = tests_dir / "research"
        prob_dst = research_root / problem.problem_id
        prob_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(problem.problem_dir, prob_dst, ignore=_IGNORE)

        problems_dir = self.root / "research" / "problems"
        parts = problem.problem_id.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            common = problems_dir / parent / "common"
            if common.is_dir():
                shutil.copytree(
                    common, research_root / parent / "common", ignore=_IGNORE
                )

        (tests_dir / "problem_config.json").write_text(
            json.dumps(
                {
                    "problem_id": problem.problem_id,
                    "ext": problem.ext,
                    "uv_project": problem.uv_project,
                    "gpu": problem.gpu,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_solution(self, task_paths: "TaskPaths", problem: ResearchProblem) -> None:
        solution_dir = task_paths.solution_dir
        if problem.reference_path and problem.reference_path.exists():
            shutil.copy2(
                problem.reference_path, solution_dir / f"reference.{problem.ext}"
            )
        else:
            LOGGER.warning("  [!] %s has no reference solution (oracle will fail)", problem.problem_id)
        solve_sh = solution_dir / "solve.sh"
        shutil.copy2(self.template_dir / "solution" / "solve.sh", solve_sh)
        solve_sh.chmod(0o755)

    def _write_task_config(self, task_paths: "TaskPaths", problem: ResearchProblem) -> None:
        template = (self.template_dir / "task.toml").read_text(encoding="utf-8")
        runtime = problem.config.get("runtime", {}) or {}
        resources = runtime.get("resources", {}) or {}
        cpus = _int_from(resources.get("cpus"), 4)
        memory_gb = _int_from(resources.get("memory"), 8)
        text = template.format(
            task_id=problem.task_id,
            problem_id=problem.problem_id,
            tag=problem.tag,
            verifier_timeout_sec=float(max(600, problem.timeout_seconds)),
            agent_timeout_sec=float(max(10800, problem.timeout_seconds)),
            build_timeout_sec=1800.0,
            cpus=cpus,
            memory_mb=memory_gb * 1024,
            storage_mb=16384,
            gpus=1 if problem.gpu else 0,
        )
        try:
            from harbor.models.task.config import TaskConfig

            config = TaskConfig.model_validate_toml(text)
            config.source = "https://github.com/FrontierCS/Frontier-CS"
            text = config.model_dump_toml()
        except ImportError:
            pass
        task_paths.config_path.write_text(text, encoding="utf-8")
