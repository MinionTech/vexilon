# Docker Quick Reference

## Fast Development Loop

### Standard Development Flow (Best for Caching)
```bash
# First time
docker compose build
docker compose up -d dev

# Iterate on code
# Edit app/main.py, app/prompts, etc.

docker compose restart dev  # ✅ Fast! (30-60s, uses hot-reload + cache)
```

### When Cache Invalidates (Expected)
```bash
# Updated dependencies
git pull  # uv.lock changed
docker compose build dev     # Rebuilds from builder stage (~3 min)
docker compose restart dev

# Updated knowledge base
# Edit app/data/labour_law/*.pdf
docker compose build dev     # Rebuilds from indexed_builder (~2 min)

# Updated Containerfile
# Edit Containerfile
docker compose build dev     # Rebuilds affected stages
```

### Nuclear Option (If Something Seems Stuck)
```bash
# Remove containers + volumes (clears everything)
docker compose down -v

# Full rebuild (takes 8-10 min, no cache)
docker compose build --no-cache
docker compose up dev
```

## Understanding Build Output

### Look for CACHED ← Good Sign
```
#8 [functional_builder 2/6] COPY PRIVACY.md ./public/docs/  CACHED
#9 [functional_builder 3/6] COPY app/.chainlit/ ./app/.chainlit/  CACHED
#10 [functional_builder 4/6] COPY app/main.py ./  RUN  ← Rebuilt (code change)
#11 [functional_builder 5/6] RUN ...                 RUN  ← Dependent rebuild
```

### Interpret Build Time
- `⚡ < 1 min`: Excellent caching (code-only changes)
- `⏱️ 1-3 min`: Good caching (data changes)
- `🐌 5-10 min`: Full rebuild (dependencies or `--no-cache`)

## Cache Invalidation Triggers

| File/Dir Changed | Stage Invalidated | Rebuild Time | Notes |
|---|---|---|---|
| `app/main.py` | functional_builder (1 layer) | ~30 sec | ✅ Fast |
| `app/prompts/` | functional_builder (1 layer) | ~30 sec | ✅ Fast |
| `app/public/` | functional_builder (1 layer) | ~30 sec | ✅ Fast |
| `app/data/labour_law/` | indexed_builder + functional | ~2 min | Uses cached FAISS if size same |
| `uv.lock` | builder + all downstream | ~3 min | Dependencies changed |
| `.git/*` | None (excluded in .dockerignore) | - | ✅ Ignored |
| `.pytest_cache/` | None (excluded in .dockerignore) | - | ✅ Ignored |
| `.pdf_cache/` | None (excluded in .dockerignore) | - | ✅ Ignored |
| `Containerfile` | Affected stage(s) only | Varies | Only rebuilds changed lines |

## Troubleshooting

### "Build takes 10 minutes every time"
1. Check `.dockerignore` is present: `ls -la .dockerignore`
2. Check for cache-busting comments in Containerfile: `grep -i "cache.buster" Containerfile`
3. Check if `uv.lock` changed: `git diff uv.lock` (rebuilds deps if yes)

### "Still slow even after optimize"
1. First full build is always slow (downloads model, builds index)
2. Try: `docker compose build --progress=plain 2>&1 | grep CACHED | wc -l`
   - If < 5 CACHED layers, something is invalidating cache
3. Check: `git status` (uncommitted files shouldn't affect Docker)

### "Volumes are huge, taking up disk space"
```bash
# Check volume sizes
docker volume ls
docker volume inspect vexilon_ollama_data

# Clean up old volumes (WARNING: deletes data)
docker volume prune
```

## Pro Tips

### Enable BuildKit (faster, better caching)
```bash
export DOCKER_BUILDKIT=1
docker compose build  # Now uses advanced caching
```

### Run tests without rebuilding app
```bash
docker compose run test-unit  # Builds test image, runs tests, exits
```

### Inspect image layers
```bash
docker history vexilon:dev  # Shows all layers and sizes
```

### Check what's in .dockerignore
```bash
cat .dockerignore
```

---

**Last Updated**: 2026-05-16  
**See Also**: `docs/DOCKER_CACHING_STRATEGY.md` for detailed architecture
