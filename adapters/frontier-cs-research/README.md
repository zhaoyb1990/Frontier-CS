# Harbor adapter — Frontier-CS research track

Generates self-contained [Harbor](https://github.com/laude-institute/harbor)
(terminal-bench) tasks from the Frontier-CS **research** track
(`research/problems/**`), one task per problem/variant.

Companion to `adapters/frontier-cs-algorithm` and `adapters/frontier-cs-2.0`.

## What it does

Each research problem already ships a self-contained evaluator
(`evaluator.py` + `evaluate.sh` + `resources/`) that scores a solution. This
adapter wraps that into a Harbor task — it does **not** re-implement scoring.

**Architecture: verifier-in-environment (no judge sidecar).** Unlike the 2.0
adapter (HTTP judge importing an `evaluate()` function), research evaluators are
CLI scripts. So the verifier runs the problem's own `evaluate.sh` directly inside
the environment container and parses the printed score. This ports the logic of
`src/frontier_cs/runner/research_docker.py` (its run-script + `_parse_score`)
minus the nested-Docker layer.

```
main container = the problem's docker image (+ python3 if the base lacks it)
  agent  : writes /app/solution.<ext>
  verify : tests/test.sh -> tests/evaluate.py runs evaluate.sh, writes the reward
```

Reward is `clamp(score / 100, 0, 1)` written to `/logs/verifier/reward.txt`.
The clamp is deliberate — research scores can exceed 100 (a kernel faster than
the reference), the same over-100 class of bug seen in algorithmic problem 186.
`score_unbounded` is preserved in `/logs/verifier/research_result.json`.

## Usage

```bash
cd adapters/frontier-cs-research

# Generate every research task into ../../datasets/frontier-cs-research/
uv run frontier-cs-research --source /path/to/Frontier-CS --overwrite

# Just one problem (P0 smoke):
uv run frontier-cs-research --source /path/to/Frontier-CS \
    --task-ids nbody_simulation/random_10k --output-dir /tmp/research_tasks --overwrite

# Override the base image (e.g. point at a mirror registry for China network):
uv run frontier-cs-research --source /path/to/Frontier-CS \
    --docker-image my-registry.example.com/triton-tlx:tlx-nv-cu122 --overwrite
```

Then run a task with Harbor (oracle = the reference solution):

```bash
harbor trial start -p datasets/frontier-cs-research/frontier-cs-research-nbody-simulation-random-10k -a oracle
```

### Oracle vs model agent

- **`-a oracle`** copies `solution/reference.<ext>` and scores it — no model, no API.
  This is the deterministic chain check.
- **`-a claude-code -m <model>`** / **`-a codex -m <model>`** run an *agentic*
  scaffold inside the container that reads `instruction.md` and writes
  `/app/solution.<ext>` over multiple turns. This needs the agent CLIs in the
  image — either bake them into your base image, or generate with `--with-agents`
  (installs **both claude-code and codex via npm**: `@anthropic-ai/claude-code`,
  `@openai/codex`):

  ```bash
  uv run frontier-cs-research --source /path/to/Frontier-CS --with-agents \
      --docker-image my-registry.example.com/triton-tlx:tlx-nv-cu122 --overwrite
  ```

Both harnesses share the problem's own `evaluate.sh`, so a given `solution.<ext>`
scores identically. They differ only in how the solution is produced: the upstream
`frontier batch research --model` does **single-shot** generation, while a Harbor
agent run is **multi-turn agentic** — so Harbor-agent scores are not directly
comparable to the upstream one-shot leaderboard.

## Generated task layout

```
frontier-cs-research-<slug>/
├── task.toml                 gpus=1 for GPU problems; cpus/mem/timeouts from config.yaml
├── instruction.md            problem readme + the Solution interface contract
├── environment/Dockerfile    FROM <problem image>; ensures python3; WORKDIR /app
├── solution/
│   ├── reference.<ext>       oracle (problem_dir/reference.* preferred)
│   └── solve.sh              copies the reference into /app/solution.<ext>
└── tests/
    ├── test.sh, evaluate.py  the verifier (runs evaluate.sh, parses score, writes reward)
    ├── problem_config.json   { problem_id, ext, uv_project, gpu }
    └── research/<id>/ ...     baked problem subtree + sibling common/ dirs
```

## Status / scope

- **127 tasks**, matching the set the leaderboard actually scores over — i.e.
  `frontier batch`'s `_get_valid_problems()` (`find research/problems -name
  evaluator.py` minus `common/resources`). Each poc_generation per-CVE case
  counts as its own problem.
  - Heads-up on an upstream inconsistency: `frontier list research` and the
    leaderboard header say "68 problems" (poc collapsed to its 4 categories), but
    the score CSVs (`batch/state.py` → `aggregate_by_model(valid_problems)`)
    average over the **127** per-case set. We follow what's scored, not the label.
    Because it's a flat per-problem average, poc's 63 cases carry ~half the
    research-track weight in that computation.
- **CPU chain — validated.** `nbody_simulation/random_10k` runs end to end (oracle
  scores, reward written, clamped).
- **GPU problems (~20 variants)** need a single NVIDIA GPU (L4/A10, 24 GB). The
  image is `andylizf/triton-tlx:tlx-nv-cu122` (`-nvcc` for qknorm). Score curves are
  calibrated for L4; other single cards run fine but the numbers aren't leaderboard-comparable.
- **poc_generation (63 tasks)** pulls vulnerability data from HuggingFace and uses
  Docker-in-Docker. Generated for parity with the scored set, but needs an HF
  mirror + DinD to run — treat as not-yet-validated.

## China / offline notes

- **Docker images** come from Docker Hub (`andylizf/triton-tlx`, `gcc:13`,
  `python:3.11-slim-trixie`). Pre-pull them or push to an Aliyun ACR and pass
  `--docker-image <mirror>` (CPU) / build with the mirrored base.
- **PyPI** (uv installs problem deps at verify time for `uv_project` problems): set
  `UV_INDEX_URL`/pip mirror in the environment.
- A few problems fetch foreign datasets at setup (`poc_generation`, `llm_router` →
  HuggingFace; `vdb_pareto` → ftp.irisa.fr). Pre-stage those or skip them initially.
- The oracle path needs **no** agent CLI (no claude.ai/npm) and, once images are
  local, **no** foreign network.

## Tests

```bash
cd adapters/frontier-cs-research
PYTHONPATH=src uv run --no-project --with pytest --with pyyaml python -m pytest tests/ -q
```

Pure-logic units (score parsing, reward clamp, task-id, language/gpu detection,
reference lookup, task generation) — no Docker required.
