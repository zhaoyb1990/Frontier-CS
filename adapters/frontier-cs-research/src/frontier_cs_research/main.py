from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .adapter import FrontierCSResearchAdapter

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[4] / "datasets" / "frontier-cs-research"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Harbor tasks from the Frontier-CS research track"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated Harbor tasks (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only generate these research problem IDs (e.g. nbody_simulation/random_10k)",
    )
    parser.add_argument(
        "--source",
        default="https://github.com/FrontierCS/Frontier-CS.git",
        help="Frontier-CS repo path or git URL (default: official upstream)",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Override the base Docker image for every task "
        "(e.g. an Aliyun ACR mirror for China network)",
    )
    parser.add_argument(
        "--with-agents",
        action="store_true",
        help="Bake claude-code + codex (both via npm) into the image for "
        "model-driven `harbor trial -a claude-code|codex` runs. Omit when the "
        "base image already bundles the agents or for oracle-only/offline use.",
    )
    args = parser.parse_args()

    source = args.source
    tmp_dir: str | None = None
    try:
        if source.startswith(("http://", "https://", "git@")):
            tmp_dir = tempfile.mkdtemp(prefix="frontier-cs-research-")
            print(f"Cloning {source}...")
            subprocess.run(["git", "clone", "--depth=1", source, tmp_dir], check=True)
            source = tmp_dir

        source_path = Path(source)
        if not (source_path / "research" / "problems").is_dir():
            raise FileNotFoundError(f"{source_path}/research/problems/ not found")

        print(f"Generating tasks -> {args.output_dir}/")
        adapter = FrontierCSResearchAdapter(
            source_path,
            args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            task_ids=args.task_ids,
            docker_image=args.docker_image,
            with_agents=args.with_agents,
        )
        results = adapter.run()
        print(f"\nDone: {len(results)} tasks generated")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
