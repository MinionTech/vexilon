# Agreement Navigator (AgNav)

The core application, documentation, and technical specifications have moved to the [app/](./app/) directory.

For the full project documentation, please see [app/README.md](./app/README.md).

---

## ⚡ Atomic Pulse Check

To verify the entire repository (Unit, Integration, E2E, and Live Smoke Tests) in a single "Grand Slam" pass, run:

```bash
podman compose --profile all up --build --abort-on-container-exit
```

This command will:
1. Build all atomic service images.
2. Run the complete test battery.
3. Start `dev` (port 7860) and `staging` (port 7861) environments.
4. Verify both live environments via the `grand-slam` master auditor.
5. Automatically shutdown all services on completion.

> [!TIP]
> **Build Efficiency:** On your very first run, use `podman compose build test-unit` to "warm" the dependency cache sequentially. This prevents parallel services from competing for the same `uv sync` lock and keeps your terminal logs clean. Subsequent runs will use the cached layers instantly.
