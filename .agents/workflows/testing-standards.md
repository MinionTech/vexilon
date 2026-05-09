---
description: Testing architecture, mocking requirements, and CI/CD testing gates
---

# Testing Standards

This document defines the testing protocols for Vexilon. **All agents must verify changes against these standards before reporting success.**

## 1. The "Mock-First" Rule

### Hugging Face (HF) Isolation
- **MANDATORY**: All unit tests must mock `get_embed_model()` and `SentenceTransformer` to prevent 3GB model downloads in CI.
- **Fixture**: Use the `mock_embedding_model` fixture in `conftest.py`.

### LLM Client Mocking
- **Fixture**: Use `mock_llm_client` for all non-integration tests. 
- **Constraint**: Mocks must support both streaming (`unified_chat_stream`) and non-streaming (`unified_chat_create`) calls.

## 2. Test Categorization

### Unit Tests (`app/tests/`)
- **Location**: `app/tests/test_*.py`
- **Focus**: UI logic, regex sanitizers, rate limiters, and chunking math.
- **Run**: `pytest app/tests/ -v`

### Integration Tests (`app/tests/integration/`)
- **Environment**: Containerized environment.
- **Focus**: Full RAG pipeline flow from query to response.
- **Run**: `podman compose up test-integration-model`

### E2E / Smoke Tests (`app/scripts/smoke_multi.py`)
- **Environment**: Containerized environment.
- **Focus**: Functional validation of the Gradio interface and final response integrity.

## 3. Coverage & Verification

### Mandatory Coverage
- **Core Logic**: `app/main.py` and `app/indexing.py` must maintain >80% coverage.
- **Report**: Coverage reports are automatically uploaded as artifacts in `pr-open.yml`.

### Deployment Integrity
- **Script**: `app/tests/test_deploy_integrity.py`
- **Rule**: This test checks for GHCR lowercasing and HF metadata compliance. Never skip this test.
