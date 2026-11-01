"""Welcome-screen starter chips: mode how-tos and high-traffic Lookup topics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chainlit as cl

from core.config import EXAMPLES, PERSONAS

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class StarterSpec:
    label: str
    message: str
    target_persona: str


GRIEVANCE_BUILDER_MESSAGE = (
    "You selected **Grievance Builder**. Switching to **Grieve** mode for strategic grievance "
    "guidance and forensic auditing—not a generic lookup.\n\n"
    "Describe what happened, which agreement articles may apply, and the remedy you are seeking."
)

MANAGEMENT_GUIDE_MESSAGE = (
    "You selected **Management Guide**. Switching to **Manage** mode for obligation-focused "
    "guidance (duty to accommodate, discipline nexus, hard cases)—not management paperwork "
    "or forms.\n\n"
    "Describe the management scenario or question you need help with."
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
        label="Grievance Builder",
        message=GRIEVANCE_BUILDER_MESSAGE,
        target_persona="Grieve",
    ),
    StarterSpec(
        label="Management Guide",
        message=MANAGEMENT_GUIDE_MESSAGE,
        target_persona="Manage",
    ),
)

STARTER_PERSONA_BY_MESSAGE: dict[str, str] = {
    spec.message: spec.target_persona for spec in STARTER_SPECS
}

STARTER_LABEL_BY_MESSAGE: dict[str, str] = {
    spec.message: spec.label for spec in STARTER_SPECS
}


def chainlit_starters() -> list[cl.Starter]:
    return [cl.Starter(label=spec.label, message=spec.message) for spec in STARTER_SPECS]


async def apply_starter_persona(message: str) -> str | None:
    """If message matches a starter chip, switch persona and return the target persona."""
    target = STARTER_PERSONA_BY_MESSAGE.get(message)
    if not target or target not in PERSONAS:
        return None
    await _sync_persona_selector(target)
    return target


async def _sync_persona_selector(persona: str) -> None:
    cl.user_session.set("persona", persona)
    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Persona",
                label="Navigator Persona",
                values=PERSONAS,
                initial_index=PERSONAS.index(persona),
            ),
        ]
    ).refresh()
