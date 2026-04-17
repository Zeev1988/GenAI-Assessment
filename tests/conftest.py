"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_sample_roots() -> Iterable[Path]:
    env = os.getenv("PHASE1_DATA_DIR")
    if env:
        yield Path(env).expanduser().resolve()

    yield REPO_ROOT / "phase1_data"
    yield REPO_ROOT.parent / "phase1_data"

    downloads = Path.home() / "Downloads"
    if downloads.exists():
        yield from downloads.glob("Home-Assignment-GenAI-KPMG*/phase1_data")


@pytest.fixture(scope="session")
def phase1_data_dir() -> Path:
    for candidate in _candidate_sample_roots():
        if candidate.is_dir():
            return candidate
    pytest.skip("phase1_data directory not found. Set PHASE1_DATA_DIR to override.")


@pytest.fixture(scope="session")
def sample_pdfs(phase1_data_dir: Path) -> list[Path]:
    pdfs = sorted(phase1_data_dir.glob("283_ex*.pdf"))
    if not pdfs:
        pytest.skip("No 283_ex*.pdf samples found in phase1_data.")
    return pdfs
