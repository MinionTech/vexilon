import asyncio
import anyio
import os
import sys
import sniffio

# The Nuclear Option: Patch sniffio directly
_orig_current_async_library = sniffio.current_async_library
def _patched_current_async_library():
    try:
        return _orig_current_async_library()
    except sniffio.AsyncLibraryNotFoundError:
        return "asyncio"

sniffio.current_async_library = _patched_current_async_library

async def main():
    print(f"Python version: {sys.version}")
    
    # We simulate the failure by running in a context where sniffio might fail
    # or just verifying it works.
    try:
        print("Testing anyio.to_thread.run_sync with sniffio patch...")
        result = await anyio.to_thread.run_sync(os.stat, ".")
        print(f"Success! result: {result.st_mode}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
