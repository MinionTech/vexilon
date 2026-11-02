"""Grievance Builder wizard state machine for Grieve persona."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WIZARD_SESSION_KEY = "grieve_wizard"

BUILDER_KICKOFF_EXACT = frozenset(
    {
        "grievance builder",
        "start grievance builder",
        "build a grievance",
        "start the grievance builder wizard",
    }
)

OPTIONAL_WIZARD_STEP_IDS = frozenset({"articles", "management_response", "witnesses_evidence"})

WIZARD_STEPS: tuple[dict[str, str], ...] = (
    {
        "id": "issue_summary",
        "title": "What happened",
        "prompt": (
            "Briefly describe the workplace issue or management action you are grieving "
            "(who, what, and why it may violate the collective agreement)."
        ),
    },
    {
        "id": "incident_date",
        "title": "Key dates",
        "prompt": (
            "What is the date of the incident or decision? "
            "Include any grievance timeline deadlines you are aware of (e.g., 30-day Step 2 limit)."
        ),
    },
    {
        "id": "articles",
        "title": "Contract articles",
        "prompt": (
            "Which collective agreement article(s) might apply (e.g., Article 8, 12, 14)? "
            "Type **skip** if you are not sure yet."
        ),
    },
    {
        "id": "management_response",
        "title": "Management response",
        "prompt": (
            "Summarize any informal discussions or written responses from management so far. "
            "Type **skip** if none yet."
        ),
    },
    {
        "id": "witnesses_evidence",
        "title": "Witnesses & evidence",
        "prompt": (
            "List witnesses, documents, or other evidence to gather (Form C fact sheet). "
            "Type **skip** if unknown."
        ),
    },
    {
        "id": "remedy",
        "title": "Remedy sought",
        "prompt": (
            "What remedy should the union seek (e.g., make whole, remove discipline, cease violation)?"
        ),
    },
)

WIZARD_INTRO = """## Grievance Builder (Grieve mode)

This wizard helps you collect facts and produce **partially filled grievance forms** and a **fact sheet** grounded in the collective agreement.

**Important:** Outputs are draft working copies for **steward review only**. They are **not** ready to file until a human steward and BCGEU staff rep review them.

Commands at any step: **restart** (start over), **exit wizard** (switch to freeform Grieve chat).

---

"""


@dataclass
class WizardTurnResult:
    handled: bool = False
    generate_draft: bool = False
    draft_query: str = ""
    assistant_message: str = ""


def default_wizard_state() -> dict[str, Any]:
    return {"active": True, "step": 0, "facts": {}, "pending_draft": False}


def is_builder_kickoff(message: str) -> bool:
    lower = (message or "").strip().lower()
    if lower in BUILDER_KICKOFF_EXACT:
        return True
    return lower.startswith("i need to file a grievance") and "builder" in lower


def wizard_state_after_failed_draft(facts: dict[str, str]) -> dict[str, Any]:
    """Retryable wizard session after draft generation failed."""
    return {
        "active": True,
        "step": len(WIZARD_STEPS),
        "facts": dict(facts),
        "pending_draft": True,
    }


def wizard_state_after_successful_draft() -> dict[str, Any]:
    return {"active": False, "step": len(WIZARD_STEPS), "facts": {}, "pending_draft": False}


def should_activate_wizard(persona: str, message: str, wizard: dict[str, Any] | None) -> bool:
    if persona != "Grieve":
        return False
    if wizard and wizard.get("active"):
        return True
    return is_builder_kickoff(message)


def _step_prompt(step_index: int) -> str:
    if step_index < 0 or step_index >= len(WIZARD_STEPS):
        return ""
    step = WIZARD_STEPS[step_index]
    return (
        f"**Step {step_index + 1} of {len(WIZARD_STEPS)} — {step['title']}**\n\n"
        f"{step['prompt']}"
    )


def wizard_intro_message() -> str:
    return WIZARD_INTRO + _step_prompt(0)


def format_facts_block(facts: dict[str, str]) -> str:
    lines = []
    for step in WIZARD_STEPS:
        value = (facts.get(step["id"]) or "").strip()
        if value and value.lower() != "skip":
            lines.append(f"- **{step['title']}:** {value}")
    return "\n".join(lines) if lines else "- (No facts collected yet)"


def build_draft_query(facts: dict[str, str]) -> str:
    facts_block = format_facts_block(facts)
    return (
        "Grievance Builder — generate draft working documents from these steward-collected facts.\n\n"
        f"{facts_block}\n\n"
        "Produce all mandatory sections from your case builder instructions, including:\n"
        "- Draft Statement of Grievance\n"
        "- Rebuttal of Management Claims (if applicable)\n"
        "- Doctrinal Logic Audit\n"
        "- Redress Sought\n"
        "- Contract & Statutory Authority with verbatim quotes\n\n"
        "Also include **### 📋 Partial Form Fields (Steward Review Required)** with subsections:\n"
        "- **Form A — Grievor Case (draft fields):** incident date, draft statement, articles, remedy\n"
        "- **Form B — Notify Designates (draft notes):** Step 1/2 notification timeline reminders\n"
        "- **Form C — Steward Fact Sheet (draft fields):** witnesses, evidence, management response summary\n\n"
        "State clearly that these are partial drafts requiring human review before filing."
    )


def has_required_facts(facts: dict[str, str]) -> bool:
    for step in WIZARD_STEPS:
        if step["id"] in OPTIONAL_WIZARD_STEP_IDS:
            continue
        val = (facts.get(step["id"]) or "").strip().lower()
        if not val or val == "skip":
            return False
    return True


def process_wizard_message(message: str, wizard: dict[str, Any] | None) -> tuple[dict[str, Any], WizardTurnResult]:
    """Advance wizard state from a user message. Pure function (no Chainlit)."""
    state = dict(wizard) if wizard else default_wizard_state()
    facts: dict[str, str] = dict(state.get("facts") or {})
    step = int(state.get("step", 0))
    msg = (message or "").strip()
    lower = msg.lower()

    if lower in {"exit wizard", "freeform", "exit"}:
        if state.get("active") or state.get("pending_draft"):
            state["active"] = False
            state["pending_draft"] = False
            return state, WizardTurnResult(
                handled=True,
                assistant_message=(
                    "Exited Grievance Builder wizard. You can continue in freeform **Grieve** mode "
                    "for forensic analysis and contract research."
                ),
            )
        return state, WizardTurnResult(handled=False)

    if lower == "restart":
        if state.get("active") or state.get("pending_draft"):
            fresh = default_wizard_state()
            return fresh, WizardTurnResult(handled=True, assistant_message=WIZARD_INTRO + _step_prompt(0))
        return state, WizardTurnResult(handled=False)

    if not state.get("active", True):
        return state, WizardTurnResult(handled=False)

    if step >= len(WIZARD_STEPS):
        return state, WizardTurnResult(handled=False)

    step_def = WIZARD_STEPS[step]
    if step == 0 and is_builder_kickoff(msg):
        return state, WizardTurnResult(
            handled=True,
            assistant_message=_step_prompt(0),
        )

    if lower == "skip":
        if step_def["id"] not in OPTIONAL_WIZARD_STEP_IDS:
            return state, WizardTurnResult(
                handled=True,
                assistant_message=(
                    f"**{step_def['title']}** is required for a useful draft. "
                    "Please provide an answer, or type **exit wizard** for freeform chat."
                ),
            )
        facts[step_def["id"]] = "skip"
        step += 1
    else:
        if not msg:
            return state, WizardTurnResult(
                handled=True,
                assistant_message="Please enter a response, **skip** (where allowed), **restart**, or **exit wizard**.",
            )
        facts[step_def["id"]] = msg
        step += 1

    state["facts"] = facts
    state["step"] = step

    if step >= len(WIZARD_STEPS):
        if not has_required_facts(facts):
            state["facts"] = facts
            state["step"] = 0
            return state, WizardTurnResult(
                handled=True,
                assistant_message=(
                    "Required facts are missing for a draft. "
                    + _step_prompt(0)
                ),
            )
        state["active"] = False
        return state, WizardTurnResult(
            handled=True,
            generate_draft=True,
            draft_query=build_draft_query(facts),
            assistant_message=(
                "✓ Fact intake complete. Generating **partial forms and fact sheet** "
                "(draft for steward review—not ready to file)…"
            ),
        )

    return state, WizardTurnResult(handled=True, assistant_message=_step_prompt(step))
