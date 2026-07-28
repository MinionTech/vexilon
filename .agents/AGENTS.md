# Custom Workspace Rules (vexilon)

## 1. Tone & Terminology Constraints
* **No Premature Finality:** NEVER refer to active branch code, patches, PRs, or build artifacts as "final" or "finalized". Always use "candidate", "latest draft", "patch candidate", or "proposal" until the changes are merged into `main`, deployed, and manually verified.

## 2. Branching & Isolation
- **Feature Branches**: All work MUST be performed in feature branches. 
- **Clean Base**: Always branch from a fresh, updated `main`. Never branch from an existing feature branch.
- **Workflow**: `git fetch origin main && git checkout -b feat/<name> origin/main`.
- **Commits**: Always use Conventional Commits.

## 3. Mandatory Verification (Testing & Linting)
- **Logic Testing**: Work is NOT ready to push or commit if it hasn't been tested. You must run the code and verify the output.
- **ShellCheck**: Every shell script (`*.sh`) MUST pass `shellcheck` verification. If `shellcheck` is not found, perform a manual line-by-line audit against ShellCheck rules.
- **No Orphans**: Scan for regressions and orphaned logic before completing tasks.

## 4. Definition of Done
Work is only marked "Done" once the following are verified via terminal:
1.  **Tested**: Functional verification successful.
2.  **Committed**: Standard Conventional Commits used.
3.  **Pushed**: Remote origin matches local state.
4.  **PR Created**: A Pull Request is open and linked in the chat.

## 5. Communication & Integrity
- **NO BULLSHITTING**: Never claim success or make unverified claims. If you haven't run the test, do not say it works.
- **ZERO GUESSING**: If you are uncertain about an API, a path, a regex, or a tool’s behavior, verify it via inspection or documentation search. Never guess.
- **Proof-of-Work**: Every turn with an edit MUST end with a verification check (e.g., a test run or linting command).

## 6. Hard Stops
- **NEVER** report "Done" without terminal verification.
- **NEVER** push code that fails linting or basic functional tests.
- **ALWAYS** stop on the first error and fix the root cause before proceeding.

## 7. Technology Constraints
To prevent accidental regression and "downgrades" that frustrate the team, the following versions are NON-NEGOTIABLE:
- **Python**: MUST remain at **3.14**. Any version other than 3.14 (including 3.12, 3.13, or lower) is FORBIDDEN.
- **LLM Models**: **Qwen 3** is the primary target. Do not use Qwen 2.5, 2.9, or any non-Qwen 3 model in production configurations.
- **Exemptions**: Lightweight models for CI/testing (e.g., `tinyllama` in `compose.yml`) are EXEMPT from the model regression rule, but the core application logic must target Qwen 3.
- **Verification**: Any change affecting `compose.yml`, `Containerfile`, or `pyproject.toml` MUST be double-checked against these constraints.

## 8. Verification Protocol
- **Incremental Verification (The "Dev" Check)**: 
  - For iterative work, use container dev verification (e.g. `podman compose up --build dev`) to verify logs and manual output.
  - **The Log Receipt**: If you claim a feature is running, you must show the relevant log lines from the container output.
- **Pre-Flight Verification**:
  - Before declaring a task "Done" or opening a PR, the agent MUST attempt to run specific test suites (e.g., `pytest app/tests/`) relevant to the change.
- **No Psychic Claims**: Never assume code works because it "looks correct." If there is no terminal output, the verification didn't happen.

## 9. Dependency Discipline & Anti-Regression Law
- **Forward-Only Rule**: NEVER pin an old version or downgrade a package to "fix" a Python 3.14 compatibility issue.
- **Upgrade First**: Dependencies MUST target current stable versions. ALWAYS attempt to upgrade first before exploring hacks or pinning.
- **NO DOWNGRADING WITHOUT DISCUSSION**: Agents are strictly forbidden from decreasing any version number (runtime, dependency, or GitHub Action version) unless explicitly directed by the user.

## 10. Implementation Restraint
- **Minimal Overhaul Rule**: If a "basic" or "minimal" overhaul is requested, agents MUST NOT add new navigation components (sidebars, tabs, menus) or change established layouts without explicit, itemized confirmation.
- **Discussion First**: Proposed UI features or aesthetic improvements MUST be presented as options first before implementing.
- **Respect User Preferences**: User-stated preferences for specific UI elements are immutable constraints.
