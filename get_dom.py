import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:7860/")
        await page.wait_for_selector('body', timeout=10000)
        await page.wait_for_timeout(5000) # Wait for React to fully mount
        
        inputs = await page.query_selector_all('textarea, input, [contenteditable="true"]')
        print(f"Found {len(inputs)} input candidates:")
        for i in inputs:
            html = await i.evaluate('el => el.outerHTML')
            print(html)
            
        print("\nSearching for any element containing 'Type your message' or similar:")
        containers = await page.query_selector_all('*')
        for c in containers:
            try:
                placeholder = await c.evaluate('el => el.getAttribute("placeholder")')
                if placeholder and "message" in placeholder.lower():
                    html = await c.evaluate('el => el.outerHTML')
                    print(f"Found by placeholder: {html}")
            except Exception:
                pass
            
        await browser.close()

asyncio.run(main())
