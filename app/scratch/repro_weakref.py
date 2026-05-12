import asyncio
import anyio
import anyio._backends._asyncio
import os
import sys

# Reproduction of the NoneType weakref error
async def test_error():
    print(f"Python version: {sys.version}")
    
    # We need to simulate being in a state where current_task() is None
    # but we are running a coroutine. This happens in 3.14 sometimes 
    # during initialization or in certain middleware.
    
    # In this script, current_task() will be main().
    # To truly reproduce, we might need to call into anyio internals.
    
    try:
        from anyio._backends._asyncio import CancelScope
        print("Testing CancelScope with None task...")
        # This is what fails in anyio: it tries to use the current task as a key
        with CancelScope():
            print("  Inside scope")
    except Exception as e:
        print(f"Caught: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_error())
