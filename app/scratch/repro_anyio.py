import asyncio
import anyio
import os
import sys

async def main():
    print(f"Python version: {sys.version}")
    try:
        # This is what Starlette calls that fails
        print("Testing anyio.to_thread.run_sync...")
        result = await anyio.to_thread.run_sync(os.stat, ".")
        print(f"Success! Stat result: {result.st_mode}")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
