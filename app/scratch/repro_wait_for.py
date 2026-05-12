"""Verify _safe_wait_for works outside a task context."""
import asyncio

# Reimplement the safe wait_for
async def _safe_wait_for(fut, timeout, **kwargs):
    loop = asyncio.get_running_loop()
    waiter = asyncio.ensure_future(fut, loop=loop)
    timeout_handle = loop.call_later(timeout, waiter.cancel)
    try:
        return await waiter
    except asyncio.CancelledError:
        raise asyncio.TimeoutError()
    finally:
        timeout_handle.cancel()

async def main():
    print(f"current_task inside main: {asyncio.current_task()}")

    # Test 1: timeout fires (event never set)
    event = asyncio.Event()
    try:
        await _safe_wait_for(event.wait(), timeout=0.1)
        print("FAIL: should have timed out")
    except asyncio.TimeoutError:
        print("PASS: timeout fired correctly")

    # Test 2: event set before timeout
    event2 = asyncio.Event()
    async def set_soon():
        await asyncio.sleep(0.05)
        event2.set()
    asyncio.ensure_future(set_soon())
    try:
        await _safe_wait_for(event2.wait(), timeout=1.0)
        print("PASS: event resolved before timeout")
    except asyncio.TimeoutError:
        print("FAIL: should not have timed out")

if __name__ == "__main__":
    asyncio.run(main())
