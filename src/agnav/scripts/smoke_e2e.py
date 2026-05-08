import asyncio
import logging
import sys
from app import startup, rag_stream

# Configure logging to see what's happening during the smoke test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke-e2e")

async def run_test():
    try:
        logger.info("Starting AgNav startup...")
        startup()
        
        logger.info("Sending E2E query: 'What is the nexus test?'")
        tokens = []
        async for token, _ in rag_stream("What is the nexus test?", []):
            if token:
                tokens.append(token)
        
        answer = "".join(tokens)
        logger.info(f"Received answer (length {len(answer)})")
        
        if len(answer) < 10:
            logger.error("FAILURE: Response too short. LLM might not be responding correctly.")
            sys.exit(1)
            
        logger.info("SUCCESS: E2E Smoke Test Passed!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILURE: An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_test())
