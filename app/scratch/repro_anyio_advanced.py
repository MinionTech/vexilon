import asyncio
import anyio
import os
import sys
import nest_asyncio

# Apply nest_asyncio as we do in main.py
nest_asyncio.apply()

async def run_stat():
    print("  Attempting run_sync...")
    try:
        result = await anyio.to_thread.run_sync(os.stat, ".")
        print(f"  Success! result: {result.st_mode}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print(f"Python version: {sys.version}")
    
    print("Testing anyio.to_thread.run_sync in main task...")
    await run_stat()
    
    print("\nTesting in a background task...")
    task = asyncio.create_task(run_stat())
    await task

if __name__ == "__main__":
    asyncio.run(main())
