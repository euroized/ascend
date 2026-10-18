"""Named stimulus and readout neurons (data/stimuli_783.json) and FlyWire ID <-> index lookup."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import data

STIMULI_783 = data.REPO_ROOT / "data" / "stimuli_783.json"
STIMULI_630 = data.REPO_ROOT / "data" / "stimuli_630.json"

# Composite stimuli, as used by Shiu et al. 2024 (figure 5: all Johnston's organ neurons).
COMPOSITES = {"jon_all": ["jon_ce", "jon_f", "jon_dm"]}


@lru_cache
def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def table(version: int = 783) -> dict:
    return _load(STIMULI_783 if version == 783 else STIMULI_630)


def stimulus(name: str, version: int = 783) -> list[int]:
    """FlyWire root IDs of a named stimulus set. 'none' is the empty set."""
    if name == "none":
        return []
    sets = table(version)["stimuli"]
    if name in COMPOSITES:
        return [i for part in COMPOSITES[name] for i in sets[part]]
    return list(sets[name])


def readouts(version: int = 783) -> dict[str, int]:
    return dict(table(version)["readouts"])


def id_to_index(path_comp: Path) -> dict[int, int]:
    """FlyWire root ID -> model neuron index (row order of the completeness file)."""
    return {int(f): i for i, f in enumerate(pd.read_csv(path_comp, index_col=0).index)}
