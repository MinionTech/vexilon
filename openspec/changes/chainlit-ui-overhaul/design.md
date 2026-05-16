# Design: Chainlit UI Overhaul

## Architecture Overview
The UI will be restructured to leverage native Chainlit 2.x components while maintaining the established navigation patterns preferred by the user.

## UI Components

### 1. Knowledge Base (Readme Tab)
- **Implementation**: Native Chainlit `chainlit.md` file.
- **Content**: Forensic links to labour law documents.
- **Behavior**: Standard "Readme" tab navigation.

### 2. Forensic Starters
- **Updated Starters**:
    - **"Discipline Analysis"**: `What are the Article 14 (Discipline) requirements for just cause?`
    - **"Grievance Builder"**: `I need to file a grievance for a member. What steps should I take?`
    - **"Steward Rights"**: `What are my rights as a steward during an investigation meeting?`
- **Benefit**: High-impact entry points for labor forensic tasks.

### 3. Integrated Mode Toggles
- **Implementation**: `cl.Action` buttons for persona switching (Lookup, Grieve, Audit, Manage).
- **Callback**: `@cl.action_callback` to update session state.
- **Benefit**: Faster switching than the gear icon menu.

### 4. Session Controls
- **Implementation**: `cl.Action` buttons for "Export Session" and "Clear Session."
- **Benefit**: Explicit user control over the ephemeral data lifecycle (PIPA compliant).

## Python 3.14 Compatibility Layer
- Consolidate patches into `app/patches.py` to keep `main.py` clean.
