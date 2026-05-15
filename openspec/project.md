# Vexilon Project Specification

## Technology Stack
- **Core**: Python 3.14 (Bleeding edge/experimental)
- **UI**: Chainlit (Transitioning from Gradio)
- **LLM Engine**: Ollama (Local)
- **Target Model**: Qwen 3 (14b or higher)
- **Package Manager**: uv

## Non-Negotiable Constraints
- **NO DOWNGRADING**: Python version must stay at 3.14.
- **NO MODEL REGRESSION**: Always use Qwen 3 in configurations.
- **Containerization**: Use the provided Containerfile and compose.yml.
