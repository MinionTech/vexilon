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


GRIEVANCE_BUILDER_GUIDE = (
    "You selected **Grievance Builder**. You're now in **Grieve** mode for strategic "
    "grievance guidance and forensic auditing—not a generic lookup.\n\n"
    "Describe what happened, which agreement articles may apply, and the remedy you "
    "are seeking."
)

MANAGEMENT_GUIDE_TEXT = (
    "You selected **Management Guide**. You're now in **Manage** mode for "
    "obligation-focused guidance (duty to accommodate, discipline nexus, hard "
    "cases)—not management paperwork or forms.\n\n"
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
        message="Grievance Builder",
        target_persona="Grieve",
        informational=True,
        assistant_reply=GRIEVANCE_BUILDER_GUIDE,
    ),
    StarterSpec(
        label="Management Guide",
        message="Management Guide",
        target_persona="Manage",
        informational=True,
        assistant_reply=MANAGEMENT_GUIDE_TEXT,
    ),
)

for _spec in STARTER_SPECS:
    if _spec.target_persona not in PERSONAS:
        raise ValueError(
            f"Starter '{_spec.label}' has invalid target_persona '{_spec.target_persona}'"
        )
    if _spec.informational and not (_spec.assistant_reply or "").strip():
        raise ValueError(f"Informational starter '{_spec.label}' requires assistant_reply")

STARTER_BY_MESSAGE: dict[str, StarterSpec] = {spec.message: spec for spec in STARTER_SPECS}


def build_persona_settings(initial_index: int = 0) -> cl.ChatSettings:
    return cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Persona",
                label="Navigator Persona",
                values=PERSONAS,
                initial_index=initial_index,
            ),
        ]
    )


def chainlit_starters() -> list[cl.Starter]:
    return [cl.Starter(label=spec.label, message=spec.message) for spec in STARTER_SPECS]


def resolve_starter(message: str) -> StarterSpec | None:
    return STARTER_BY_MESSAGE.get(message)


async def apply_starter_persona(message: str) -> StarterSpec | None:
    """If message matches a starter chip, switch persona when needed."""
    spec = resolve_starter(message)
    if not spec:
        return None

    current = cl.user_session.get("persona") or "Lookup"
    if spec.target_persona != current:
        try:
            await _sync_persona_selector(spec.target_persona)
        except Exception as exc:
            logger.warning("[starters] persona sync failed for '%s': %s", spec.label, exc)
            cl.user_session.set("persona", spec.target_persona)
    return spec


async def _sync_persona_selector(persona: str) -> None:
    cl.user_session.set("persona", persona)
    await build_persona_settings(PERSONAS.index(persona)).refresh()
