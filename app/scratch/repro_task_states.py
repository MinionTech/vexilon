"""
Reproduces the _task_states KeyError and verifies the bypass fix.
Run with: uv run python scratch/repro_task_states.py
"""
import asyncio
import os
import anyio.to_thread
import anyio._backends._asyncio as _anyio_asyncio_backend

_orig_run_sync = anyio.to_thread.run_sync

async def _patched_run_sync(func, *args, abandon_on_cancel=False, limiter=None):
    current = asyncio.current_task()
    if current is None or current not in _anyio_asyncio_backend._task_states:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)
    return await _orig_run_sync(func, *args, abandon_on_cancel=abandon_on_cancel, limiter=limiter)

anyio.to_thread.run_sync = _patched_run_sync

async def main():
    current = asyncio.current_task()
    in_task_states = current in _anyio_asyncio_backend._task_states if current else False
    print(f"current_task: {current}")
    print(f"in _task_states: {in_task_states}")
    print(f"_task_states count: {len(_anyio_asyncio_backend._task_states)}")

    # This is exactly what Starlette's FileResponse does
    result = await anyio.to_thread.run_sync(os.stat, ".")
    print(f"PASS: os.stat result mode={result.st_mode}")

if __name__ == "__main__":
    asyncio.run(main())
