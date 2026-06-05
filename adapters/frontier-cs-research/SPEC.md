# frontier-cs-research adapter — Spec

Generate self-contained Harbor (terminal-bench) tasks from the Frontier-CS
**research** track (`research/problems/**`), mirroring `frontier-cs-2.0`.

Each research problem already ships a self-contained Python evaluator
(`evaluator.py` + `evaluate.sh` + `resources/`) that, given a solution, prints a
score. We wrap that — we do **not** re-implement scoring.

## Architecture: verifier-in-environment (no judge sidecar)

Unlike `frontier-cs-2.0` (HTTP judge_server importing an `evaluate()` function),
research evaluators are **CLI scripts** (`argparse` `main`). So the verifier runs
the problem's own `evaluate.sh` directly inside the environment container and
parses the score. This ports `src/frontier_cs/runner/research_docker.py` logic
(its run-script + `_parse_score`) minus the nested-Docker layer.

```
main container = the problem's docker image (+ python3 if missing)
  agent phase  : writes /app/solution.<ext>
  verify phase : tests/test.sh -> tests/evaluate.py runs evaluate.sh, writes reward
```

## Generated task layout (per problem)

```
frontier-cs-research-<slug>/
├── task.toml                       gpus = 1 if problem needs GPU else 0; timeouts/cpus/mem from config
├── instruction.md                  problem readme + "write /app/solution.<ext> implementing Solution"
├── environment/
│   └── Dockerfile                  FROM <problem_image>; ensure python3; WORKDIR /app
├── solution/
│   ├── reference.<ext>             oracle (problem_dir/reference.<ext> preferred)
│   └── solve.sh                    cp /solution/reference.<ext> -> /app/solution.<ext>
└── tests/
    ├── test.sh                     mkdir /logs/verifier; python3 /tests/evaluate.py
    ├── evaluate.py                 verifier (below)
    ├── problem_config.json         { problem_id, ext, uv_project, gpu }
    └── research/                   baked problem subtree (mirrors research_docker workspace)
        └── <problem_id>/ ...       evaluator.py, evaluate.sh, resources/, + parent .../common/
```

## Verifier contract (tests/evaluate.py)

Runs inside the env container. Steps (mirror research_docker):
1. Locate `/app/solution.<ext>` (ext from `problem_config.json`).
2. Build `/work` workspace: copy `tests/research` → `/work/research`; copy solution →
   `/work/execution_env/solution_env/solution.<ext>` (path that `evaluate.sh` hardcodes).
3. `cd` to the baked problem dir (the one containing `evaluator.py`), best-effort
   `uv pip install` of `uv_project` deps (only if present; tolerate offline), run
   `set_up_env.sh` if present, then run `evaluate.sh`.
4. `parse_score(stdout+stderr)`: last numeric line → `score [score_unbounded]` (0–100).
5. `reward = clamp(score/100, 0, 1)`. Write Harbor reward files.

### Reward files (Harbor contract)
- `/logs/verifier/reward.txt`  — single float in `[0,1]` (Harbor reads this)
- `/logs/verifier/reward.json` — `{"reward": <float>}`
- `/logs/verifier/research_result.json` — sidecar: score, score_unbounded, detail, logs tail

**Clamp to [0,1] is mandatory** — research scores can exceed 100 (speedup beats
reference, same class of bug as algorithmic-186). `score_unbounded` is preserved
in the sidecar for visibility.

## Pure functions to TDD (no Docker)
- `parse_score(text) -> (score|None, score_unbounded|None, err|None)` — last-numeric-line, skips `[`/INFO/ERROR lines.
- `clamp_reward(x) -> float in [0,1]`.
- `normalize_task_id(problem_id) -> "frontier-cs-research-<slug>"`.
- `detect_language(problem_dir, config) -> ("python"|"cpp", ext)`.
- `detect_gpu(config) -> bool` (docker.gpu | runtime.requires_gpu | resources.accelerators).
- `find_reference(problem_dir, solutions_root, problem_id, ext) -> Path|None`.
- `FrontierCSResearchAdapter.generate_task` — smoke test: nbody → assert file structure + content.

## Scope / phases
- P0: nbody_simulation/random_10k (CPU, gcc:13, no datasets) — oracle scores end to end.
- P1: all CPU problems (some need dataset baking).
- P2: GPU problems (triton image, gpus=1) — needs A10/L4.

## China-friendly defaults
- `--docker-image` / image override so the base image can point to an Aliyun ACR mirror.
- Oracle path needs **no** claude.ai/npm (no agent CLI baked) and **no** foreign net once the image is local.
