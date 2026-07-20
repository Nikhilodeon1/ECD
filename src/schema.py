"""Shared case schema for both data sources.

This is the "clean" pre-adversarial case format for Week 4. Adversarial notes
(category, severity, adversarial_note) get attached to these in Week 6 per the
taxonomy in docs/week3-taxonomy-and-vignettes.md.
"""
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BaseCase:
    id: str
    source: str  # "medqa" or "mimic-iv-note"
    case_summary: str  # the presenting case / history used as evidence
    original_note: str  # the note text a model would actually be shown
    diagnosis_ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
