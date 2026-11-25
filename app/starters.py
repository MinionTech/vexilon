"""Welcome-screen starter chips: mode how-tos and high-traffic Lookup topics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chainlit as cl

from core.config import EXAMPLES, PERSONAS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StarterSpec:
    label: str
    message: str
    target_persona: str
    informational: bool = False
    assistant_reply: str | None = None
    starts_wizard: bool = False


GRIEVANCE_BUILDER_GUIDE = (
    "You selected **Grievance Builder** for strategic grievance guidance and "
    "forensic auditing—not a generic lookup.\n\n"
    "Describe what happened, which agreement articles may apply, and the remedy you "
    "are seeking."
)

STARTER_SPECS: tuple[StarterSpec, ...] = (
    StarterSpec(
        label="Discipline Just Cause",
        message=EXAMPLES[0],
        target_persona="Lookup",
    ),
    StarterSpec(
        label="Steward Rights",
        message=EXAMPLES[1],
        target_persona="Lookup",
    ),
    StarterSpec(
        label="Benefits",
        message="What extended health and dental benefits are covered under my plan?",
        target_persona="Lookup",
    ),
    StarterSpec(
        label="Grievance Builder",
        message="Grievance Builder",
        target_persona="Lookup",
        informational=True,
        assistant_reply=GRIEVANCE_BUILDER_GUIDE,
        starts_wizard=True,
    ),
)

for _spec in STARTER_SPECS:
    if _spec.target_persona not in PERSONAS:
        raise ValueError(
            f"Starter '{_spec.label}' has invalid target_persona '{_spec.target_persona}'"
        )
    if _spec.informational and not (_spec.assistant_reply or "").strip():
        raise ValueError(f"Informational starter '{_spec.label}' requires assistant_reply")
    if _spec.starts_wizard and not _spec.informational:
        raise ValueError(
            f"Starter '{_spec.label}' starts a wizard but is not informational"
        )

STARTER_BY_MESSAGE: dict[str, StarterSpec] = {spec.message: spec for spec in STARTER_SPECS}


def chainlit_starters() -> list[cl.Starter]:
    return [cl.Starter(label=spec.label, message=spec.message) for spec in STARTER_SPECS]


def resolve_starter(message: str) -> StarterSpec | None:
    return STARTER_BY_MESSAGE.get(message)


async def apply_starter_persona(message: str) -> StarterSpec | None:
    """If message matches a starter chip, return its StarterSpec."""
    return resolve_starter(message)
