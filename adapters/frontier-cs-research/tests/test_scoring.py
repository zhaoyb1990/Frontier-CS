"""TDD: the verifier's pure scoring helpers (loaded from the shipped template).

We import the *actual* template tests/evaluate.py by path so the code that runs
in the container is exactly what we test (no drift).
"""
import importlib.util
from pathlib import Path

import pytest

TEMPLATE_EVALUATE = (
    Path(__file__).resolve().parents[1]
    / "src" / "frontier_cs_research" / "task-template" / "tests" / "evaluate.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("research_verifier", TEMPLATE_EVALUATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verifier = _load()


# ---- parse_score -----------------------------------------------------------

def test_parse_two_numbers_last_line():
    assert verifier.parse_score("compiling...\nrunning\n83.5 91.2") == (83.5, 91.2, None)


def test_parse_single_number_duplicates_unbounded():
    s, su, err = verifier.parse_score("noise\n42")
    assert (s, su, err) == (42.0, 42.0, None)


def test_parse_skips_log_and_score_label_lines():
    # last real numeric line wins; "Score: x/100" and "[..]" / INFO lines skipped
    text = "[setup] go\nINFO ready\nScore: 83.5/100\n83.5 91.2"
    assert verifier.parse_score(text) == (83.5, 91.2, None)


def test_parse_score_can_exceed_100_unclamped():
    # parser does NOT clamp; clamp happens in reward mapping
    assert verifier.parse_score("150.0 150.0") == (150.0, 150.0, None)


def test_parse_error_when_no_number():
    s, su, err = verifier.parse_score("Traceback (most recent call last):\nERROR: boom")
    assert s is None and su is None and err is not None


def test_parse_error_on_empty():
    s, su, err = verifier.parse_score("   \n  ")
    assert s is None and su is None and err is not None


# ---- clamp_reward ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (1.5, 1.0),      # speedup beats reference -> still capped (186 lesson)
    (-0.2, 0.0),
    (0.835, 0.835),
    (0.0, 0.0),
    (1.0, 1.0),
])
def test_clamp_reward(raw, expected):
    assert verifier.clamp_reward(raw) == expected
