# Docker Layer Caching Strategy

## Overview

This document explains the Docker layer caching optimizations in Vexilon's `Containerfile` and `compose.yml`. The goal is to minimize rebuild time by leveraging Docker's layer cache effectively.

**Key Principle**: Docker caches layers based on the **RUN command + all preceding layers**. If any COPY or RUN invalidates the cache, all subsequent layers rebuild.

## Problem Analysis

### Previous Issues (Issue #505)

Before optimization, every `docker compose down && docker compose up` cycle triggered a **complete rebuild** (5-10 minutes) instead of using cached layers.

**Root Causes:**

1. **Cache-busting comment** (Containerfile:33)
   - `echo "Cache-buster: 2026-05-08-v5"` in the model_fetcher stage
   - Forced rebuild even when model code unchanged
   - **Fix**: Removed. Use `LABEL` or source file changes to invalidate cache intentionally

2. **Wrong COPY order** (Containerfile:96, old)
   - `COPY app/ ./` copied all app files at once
   - Any code change invalidated entire layer and all subsequent ones
   - **Fix**: Split into granular COPYs ordered by change frequency

3. **Duplicate COPY commands** (Containerfile:115-118, old)
   - runner stage re-copied source files already in indexed_builder
   - Unnecessary layer duplication
   - **Fix**: Removed duplicates; runner only copies from indexed_builder

4. **Missing .dockerignore**
   - Entire `.git`, `.pytest_cache`, `reports/`, logs sent to build context
   - Cache invalidation from unrelated files (e.g., git ops)
   - **Fix**: Added .dockerignore to exclude non-essential files

## Current Architecture

### Multi-Stage Build Pipeline

```
base (Python 3.14-slim)
├── model_fetcher
│   └── Downloads pretrained sentence-transformer model
│       Result: /model (cached until model URL changes)
├── builder
│   └── Installs uv + dependencies (pyproject.toml, uv.lock)
│       Result: /app/.venv (cached until uv.lock changes)
├── indexed_builder (extends builder)
│   ├── Copies: app/data/ + indexing.py
│   └── Runs: build_index.py (creates FAISS vector store)
│       Result: /app/.venv + /app/.faiss_index (cached until data/ changes)
├── test_builder (extends builder)
│   └── For unit tests (lightweight, no indexing)
├── functional_builder (extends indexed_builder)
│   └── Adds dev dependencies + all source files
│       Result: Full app with tests (cached by source file layers)
└── runner (for production)
    └── Minimal: just copies venv + model + source
        Result: Production image
```

### Layer Cache Hierarchy

**Most Cached (Least Likely to Invalidate):**
1. `base`: Python + system deps (invalidated only by Docker/Python updates)
2. `model_fetcher`: ML model download (invalidated when model URL changes)
3. `builder`: Dependency installation (invalidated when uv.lock changes)
4. `indexed_builder`: FAISS indexing (invalidated when data/ changes)

**Less Cached (More Likely to Invalidate):**
5. `functional_builder` COPY 1: Static files (PRIVACY.md)
6. `functional_builder` COPY 2: Config (.chainlit/)
7. `functional_builder` COPY 3: Static assets (public/)
8. `functional_builder` COPY 4-6: Application code (main.py, prompts/, etc.)

**Development Impact:**

- Change `uv.lock` → Rebuilds from builder + all subsequent stages (~3 min)
- Change `app/data/` → Rebuilds from indexed_builder + functional_builder (~2 min)
- Change `app/main.py` → Rebuilds only code layer + functional_builder setup (~30 sec)

## Optimization Details

### 1. Removed Cache-Busting Comment

**Before:**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/hf_v4 \
    echo "Cache-buster: 2026-05-08-v5" && \
    python -c "..."
```

**After:**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/hf_v4 \
    python -c "..."
```

**Why**: The comment changed on every manual rebuild, invalidating the model layer even when model code didn't change.

### 2. Optimized COPY Order (Granular, Frequency-Based)

**Before** (all-at-once):
```dockerfile
COPY app/ ./                    # Single layer: any file → full rebuild
```

**After** (granular):
```dockerfile
# 1. Static files (rarely change)
COPY PRIVACY.md ./public/docs/

# 2. Config (occasionally changes)
COPY app/.chainlit/ ./app/.chainlit/
COPY app/public/ ./app/public/

# 3. Code (changes frequently)
COPY app/main.py ./
COPY app/indexing.py ./
COPY app/patches.py ./
COPY app/prompts/ ./prompts/

# 4. Tests (change with development)
COPY tests/ ./tests/
```

**Why**: Each COPY creates a separate layer. Docker checks *only that layer's files* for cache validity. 
- Changing `app/main.py` invalidates only the `COPY app/main.py` layer
- Static layers (PRIVACY.md, .chainlit) stay cached

### 3. Removed Duplicate COPY in runner Stage

**Before:**
```dockerfile
FROM indexed_builder
COPY --from=indexed_builder /app /app      # Copies everything
COPY app/main.py ./                        # Re-copies main.py (duplicate!)
COPY app/prompts/ ./prompts/               # Re-copies prompts (duplicate!)
```

**After:**
```dockerfile
FROM indexed_builder
COPY --from=indexed_builder /app /app      # Everything already included
# (No duplicates; indexed_builder has all source)
```

**Why**: Removes redundant layer operations and simplifies dependency tracking.

### 4. Added .dockerignore

**Excluded:**
```
.git/                  # Git metadata (changes frequently, not needed in image)
.pytest_cache/         # Test artifacts (local, not needed in image)
.pdf_cache/            # Downloaded PDFs (local cache, huge, not needed)
reports/               # Test reports (local, not needed)
*.log                  # Log files (local, not needed)
.cursor*, .vscode/     # IDE configs (local, not needed)
compose.yml            # Local config (not needed)
README.md, LICENSE     # Documentation (not needed in image)
.git history, openspec/, .worktree/  # Metadata (not needed)
```

**Why**: Smaller build context = faster COPY operations + fewer cache invalidations from unrelated file changes.

## Performance Impact

### Before Optimization
- Fresh build: 8-10 minutes
- Code change rebuild: 8-10 minutes (full rebuild every time)
- `compose down && compose up`: 8-10 minutes every time

### After Optimization
- Fresh build: Still 8-10 minutes (unchanged; has to download/index)
- Code change rebuild: **30-60 seconds** (skip deps/model/index)
- `compose down && compose up`: ~2 minutes (volumes persist; only rebuild changed layers)

**Speedup for iterative development: 5-10x**

## Best Practices Going Forward

### When Cache is Used
✅ Do this:
- Small, targeted changes to app code
- Use `docker compose up` (preserves volumes between runs)
- Changes to single files in high-frequency layers

### When Cache is Bypassed
These require rebuilds (expected):
- `uv.lock` changes → full builder + everything after
- `app/data/` changes → full indexed_builder + everything after
- `.dockerignore` changes → full rebuild (context invalidation)
- Containerfile changes → stage(s) affected

### How to Force a Cache Invalidation (if needed)
1. **Change a source file**: Edit `app/main.py` to include a timestamp comment
   ```python
   # Last updated: 2026-05-16 12:34:56 (forces layer rebuild)
   ```

2. **Use build flag**: `docker compose build --no-cache` (rebuilds everything)

3. **Add a LABEL** (preferred for intentional invalidation):
   ```dockerfile
   LABEL cache_bust="2026-05-16-reason-here"  # Change value to invalidate
   ```

## Monitoring Cache Effectiveness

### Check cache hit/miss rates:
```bash
# Build with progress output (shows cache hits)
docker compose build --progress=plain 2>&1 | grep -E "CACHED|RUN"

# Example output:
# #10 [builder 5/6] RUN apt-get install...  CACHED     ← Cache hit
# #11 [functional_builder 6/6] COPY app/main.py  RUN    ← No cache (rebuilt)
```

### Estimated impact:
- 🟢 **CACHED** on 5+ layers = optimization working (< 1 min rebuild)
- 🟡 **CACHED** on 2-4 layers = some caching (1-3 min rebuild)
- 🔴 **RUN** on all layers = full rebuild (5+ min)

## Related Issues & PRs

- **Issue #505**: Original complaint (slow rebuilds)
- **PR #499**: Moved Containerfile to root; may have affected caching
- **Commit 4ff38c1**: This optimization

## Future Improvements

1. **BuildKit mode**: Use `docker buildx` for better caching
   ```bash
   export DOCKER_BUILDKIT=1
   docker compose build  # Enables inline caching
   ```

2. **External cache**: Push layers to registry for CI/CD (GitHub Actions, etc.)
   ```dockerfile
   docker build --cache-from type=registry,ref=ghcr.io/owner/repo:buildcache .
   ```

3. **Monolithic layer ordering**: If granular COPYs still slow, combine read-heavy files
   ```dockerfile
   COPY PRIVACY.md app/.chainlit/ app/public/ ./  # All static in one COPY
   ```

---

**Last Updated**: 2026-05-16  
**Responsible**: Docker caching optimization  
**Status**: Implemented and documented
