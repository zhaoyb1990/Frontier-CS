#!/usr/bin/env python3
"""Verifier for Frontier-CS research Harbor tasks.

Runs inside the environment container at verify time. Executes the problem's own
`evaluate.sh` against the agent's `/app/solution.<ext>`, parses the score, and
writes the Harbor reward files. Mirrors
`src/frontier_cs/runner/research_docker.py` (its run-script + `_parse_score`)
minus the nested-Docker layer.

Self-contained on purpose: it must run in a bare problem image with no package
install. The pure helpers (`parse_score`, `clamp_reward`, `load_config`) are unit
tested directly from this file.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path("/tests")
APP_DIR = Path("/app")
WORK = Path("/work")  # evaluate.sh hardcodes /work/execution_env/... so WORK must be /work
VERIFIER_DIR = Path("/logs/verifier")
REWARD_TXT = VERIFIER_DIR / "reward.txt"
REWARD_JSON = VERIFIER_DIR / "reward.json"
RESULT_JSON = VERIFIER_DIR / "research_result.json"


def load_config(tests_dir: Path = TESTS_DIR) -> dict:
    p = tests_dir / "problem_config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def clamp_reward(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def make_scripts_executable(root) -> None:
    """Restore the +x bit on every *.sh under ``root``.

    Many upstream scripts are checked into git without execute permission, but
    evaluate.sh calls run_evaluator.sh / set_up_env.sh directly (not via `bash`),
    so they must be executable. Mirrors research_docker's
    `find /work -name '*.sh' -exec chmod +x`.
    """
    for sh in Path(root).rglob("*.sh"):
        try:
            sh.chmod(sh.stat().st_mode | 0o111)
        except OSError:
            pass


def parse_score(output: str):
    """Return (score, score_unbounded, err). Last numeric line wins; lines
    starting with '[' or containing INFO/ERROR are treated as logs. A numeric
    line is one or two floats: "<score> [<score_unbounded>]" on a 0-100 scale."""
    lines = output.strip().split("\n")
    for line in reversed(lines):
        s = line.strip()
        if not s or s.startswith("[") or "INFO" in s or "ERROR" in s:
            continue
        parts = s.split()
        try:
            score = float(parts[0])
            score_unbounded = float(parts[1]) if len(parts) > 1 else score
            return score, score_unbounded, None
        except (ValueError, IndexError):
            continue
    for line in lines:
        if "Error" in line or "ERROR" in line:
            return None, None, line.strip()
    return None, None, "Could not parse score from output"


def write_reward(reward, *, score=None, score_unbounded=None, detail="", logs="") -> None:
    reward = clamp_reward(reward)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    REWARD_TXT.write_text(str(reward))
    REWARD_JSON.write_text(json.dumps({"reward": reward}))
    RESULT_JSON.write_text(
        json.dumps(
            {
                "reward": reward,
                "score": score,
                "score_unbounded": score_unbounded,
                "detail": detail,
                "logs_tail": (logs or "")[-4000:],
            },
            indent=2,
        )
    )
    print(f"[verifier] reward={reward} score={score} score_unbounded={score_unbounded} detail={detail}")


def _run(cmd, cwd, env, **kw):
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, **kw)


def main() -> None:
    cfg = load_config()
    ext = cfg.get("ext", "py")
    uv_project = cfg.get("uv_project")

    # 1) locate the agent solution
    solution = APP_DIR / f"solution.{ext}"
    if not solution.exists():
        cands = sorted(APP_DIR.glob("solution.*"))
        if cands:
            solution = cands[0]
            ext = solution.suffix.lstrip(".")
        else:
            write_reward(0.0, detail=f"/app/solution.{ext} not found")
            return

    # 2) build /work mirroring research_docker
    WORK.mkdir(parents=True, exist_ok=True)
    dst_research = WORK / "research"
    if dst_research.exists():
        shutil.rmtree(dst_research)
    shutil.copytree(TESTS_DIR / "research", dst_research)

    make_scripts_executable(dst_research)

    exec_env = WORK / "execution_env" / "solution_env"
    exec_env.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solution, exec_env / f"solution.{ext}")

    # 3) find the problem dir (the one with evaluator.py, not common/resources)
    evals = [
        p
        for p in dst_research.rglob("evaluator.py")
        if "/common/" not in str(p) and "/resources/" not in str(p)
    ]
    if not evals:
        write_reward(0.0, detail="evaluator.py not found in baked problem tree")
        return
    problem_dir = evals[0].parent

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")

    # 4) best-effort deps for python problems (tolerate offline / preinstalled)
    if uv_project:
        proj = (problem_dir / uv_project).resolve()
        if (proj / "pyproject.toml").exists():
            if shutil.which("uv") is None:
                _run([sys.executable, "-m", "pip", "install", "-q", "uv"], problem_dir, env)
            if shutil.which("uv"):
                _run(["uv", "pip", "install", "--system", str(proj)], problem_dir, env)

    setup = problem_dir / "set_up_env.sh"
    if setup.exists():
        _run(["bash", str(setup)], problem_dir, env)

    # 5) run the problem's own evaluate.sh and capture everything
    proc = _run(["bash", "evaluate.sh"], problem_dir, env, capture_output=True)
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(logs)

    score, score_unbounded, err = parse_score(logs)
    if score is None:
        write_reward(0.0, detail=err or f"evaluate.sh exit {proc.returncode}", logs=logs)
        return
    write_reward(score / 100.0, score=score, score_unbounded=score_unbounded, detail="ok", logs=logs)


if __name__ == "__main__":
    main()
