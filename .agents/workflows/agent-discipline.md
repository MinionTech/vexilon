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

## 6. Technology Constraints
To prevent accidental regression and "downgrades" that frustrate the team, the following versions are NON-NEGOTIABLE:
- **Python**: MUST remain at **3.14**. Any version other than 3.14 (including 3.12, 3.13, or lower) is FORBIDDEN.
- **LLM Models**: **Qwen 3** is the primary target. Do not use Qwen 2.5, 2.9, or any non-Qwen 3 model in production configurations.
- **Exemptions**: Lightweight models for CI/testing (e.g., `tinyllama` in `compose.yml`) are EXEMPT from the model regression rule, but the core application logic must target Qwen 3.
- **Verification**: Any change affecting `compose.yml`, `Containerfile`, or `pyproject.toml` MUST be double-checked against these constraints.

## 7. Verification Protocol
- **Incremental Verification (The "Dev" Check)**: 
  - For iterative work, use `podman compose up --build dev` (or equivalent) to verify logs and manual output.
  - **The Log Receipt**: If you claim a feature is running, you must show the relevant log lines from the container output.
- **Final Verification (The "Pre-Flight" Check)**:
  - Before declaring a task "Done" or opening a PR, the agent MUST attempt to run specific test services (e.g., `test-unit` or `test-cache`) if they are relevant to the change.
- **Environment Awareness**: 
  - Always prefer `podman` over `docker` if available in the environment, following the project's established patterns.
- **No Psychic Claims**: Never assume code works because it "looks correct." You are not a compiler. If there is no terminal output, the verification didn't happen.
