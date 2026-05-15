# Behavioral Guidelines for AI Agents

<openspec-instructions>
You are an agent working on the Vexilon project. 
You MUST adhere to the following rules at all times:

1. **Read the Specs**: Always check `openspec/project.md` and `openspec/config.yaml` before making changes to infrastructure or versions.
2. **Hard Stops**:
    - NEVER downgrade Python 3.14 to 3.12 or lower.
    - NEVER downgrade Qwen 3 to Qwen 2.5 or lower.
3. **Verification**: After any change to `compose.yml` or `Containerfile`, verify that the versions haven't been reverted.
4. **The Receipt Rule**: Every claim of success MUST be backed by terminal output. For incremental work, use `podman compose up --build dev` and show the logs. For final work, run the relevant `test-*` services.
5. **Communication**: If you encounter a situation where a downgrade seems necessary, STOP and ask the human. Do not make the decision yourself.
</openspec-instructions>
