# Agent Behavioral Discipline

To prevent unprofessional, sloppy, and unverified work, the following rules are MANDATORY for all agentic operations. These rules take precedence over any model-internal defaults.

## 1. Branching & Isolation
- **Feature Branches**: All work MUST be performed in feature branches. 
- **Clean Base**: Always branch from a fresh, updated `main`. Never branch from an existing feature branch.
- **Workflow**: `git fetch origin main && git checkout -B feat/<name> origin/main`.
- **Commits**" Always use conventional commits.

## 2. Mandatory Verification (Testing & Linting)
- **Logic Testing**: Work is NOT ready to push or commit if it hasn't been tested. You must run the code and verify the output.
- **ShellCheck**: Every shell script (`*.sh`) MUST pass `shellcheck` verification. If `shellcheck` is not found, you must locate it or perform a manual, line-by-line forensic audit against ShellCheck rules.
- **No Orphans**: Scan for Regressions and orphaned logic before finishing.

## 3. Definition of Done
Work is only marked "Done" once the following are verified via terminal:
1.  **Tested**: Functional verification successful.
2.  **Committed**: Standard Conventional Commits used.
3.  **Pushed**: Remote origin matches local state.
4.  **PR Created**: A Pull Request is open and linked in the chat.

## 4. Communication & Integrity
- **NO BULLSHITTING**: Never claim success or make unverified claims. If you haven't run the test, do not say it works.
- **ZERO GUESSING**: If you are uncertain about an API, a path, a regex, or a tool’s behavior, verify it via `run_command` or `search_web`. Never guess.
- **Proof-of-Work**: Every turn with an edit MUST end with a verification check (e.g., a test run or linting command).

## 5. Hard Stops
- **NEVER** report "Done" without terminal verification.
- **NEVER** push code that fails linting or basic functional tests.
- **ALWAYS** stop on the first error and fix the root cause before proceeding.
