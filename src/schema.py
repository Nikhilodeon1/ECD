"""Shared case schema for MedQA and MIMIC-IV-Note cases (pre-adversarial)."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseCase:
    id: str
    source: str  # medqa | mimic-iv-note
    case_summary: str
    original_note: str  # evidence text shown to the model
    diagnosis_ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
